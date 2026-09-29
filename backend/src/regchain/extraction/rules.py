"""Deterministic multi-clause baseline; ambiguous context is never invented."""
import json
import re

from .grounding import LEGACY_MODAL, MODAL, CONDITIONS, EXCEPTIONS, binding, modality, qualifier_tails, standalone, turkish, turkish_modal

SENTENCE = re.compile(r'(?<=[.;!?])\s+(?=[A-Z])')
LABEL_PREFIX = re.compile(r'^\s*\d+[A-Z]?(?:\.\d+[A-Z]*)+\s+(?:[RG]\s+)?')


class RulesProvider:
    name = 'rules'
    model_version = 'conservative-clauses-v2'

    def generate(self,text: str) -> str:
        from .classify import duty_modals
        matches = duty_modals(text)
        empty = lambda status: json.dumps({'status':status,'obligations':[]})
        if not matches:
            return empty('NO_EXPLICIT_OBLIGATION')
        if turkish(text):
            return self.generate_turkish(text, matches, empty)
        output = []
        previous_subject = None
        for index,match in enumerate(matches):
            boundaries = [m.end() for m in SENTENCE.finditer(text[:match.start()])]
            start = boundaries[-1] if boundaries else 0
            prefix = LABEL_PREFIX.sub('',text[start:match.start()]).strip()
            if index and start < matches[index-1].end():
                connector = re.search(r'\b(?:and|or)\s*$',prefix,re.I)
                if not connector or not previous_subject:
                    return empty('INSUFFICIENT_EVIDENCE')
                # Inherit only across an explicit coordination, not another sentence.
                subject = previous_subject
            else:
                # A leading condition remains evidence, but is not part of the actor.
                subject = prefix.rsplit(',',1)[-1].strip() if re.match(r'^(if|where|when)\b',prefix,re.I) else prefix
            if (not subject or len(subject)>160 or any(c in subject for c in '.;:?!')
                or re.search(r'\b(?:we|i|our|the fca|this publication|this policy statement)\b',subject,re.I)):
                return empty('INSUFFICIENT_EVIDENCE')
            if modality(match.group()) in ('MAY','MAY_NOT') and re.match(r'\s*(?:\d|be\b|have\b|result\b|cause\b)',text[match.end():],re.I):
                return empty('INSUFFICIENT_EVIDENCE')
            end = len(text)
            following_boundaries = list(SENTENCE.finditer(text,match.end()))
            if following_boundaries:
                end = following_boundaries[0].start()
            if index+1 < len(matches) and matches[index+1].start() < end:
                between = text[match.end():matches[index+1].start()]
                connector = re.search(r'\s+(?:and|or)\s*$',between,re.I)
                if not connector:
                    return empty('INSUFFICIENT_EVIDENCE')
                end = match.end()+connector.start()
            action = text[match.end():end].strip()
            if not action:
                return empty('INSUFFICIENT_EVIDENCE')
            mode = modality(match.group())
            output.append({'source_quote':text,'subject':subject,'modality':mode,
                'required_action':None if mode.endswith('_NOT') else action,
                'prohibited_action':action if mode.endswith('_NOT') else None,
                'conditions':qualifier_tails(text,CONDITIONS),'exceptions':qualifier_tails(text,EXCEPTIONS),
                'confidence_score':'0.5000'})
            previous_subject = subject
        return json.dumps({'status':'EXTRACTED','obligations':output})

    def generate_turkish(self, text, matches, empty):
        """Subject, then action, then the modal: one duty per sentence, nothing inferred.

        A v0.18 marker the rule cannot read makes the whole paragraph INSUFFICIENT_EVIDENCE, as
        before; a v0.19 soft marker (grounding.binding) it cannot read is only left out, because
        verify does not demand it: the baseline keeps every candidate it produced before.
        """
        output = []
        # A qualifier word that is itself a soft marker ("... durumunda uygulanmaz", "... şartıyla
        # mümkündür") ends a duty's action and is not also one of its qualifiers.
        ends = {m.end() for m in matches if not binding(m)}
        own = lambda spans: [span for span in spans if text.find(span) + len(span) not in ends]
        for match in matches:
            soft = not binding(match)
            if not turkish_modal(match.group()):
                return empty('INSUFFICIENT_EVIDENCE')
            starts = [m.end() for m in re.finditer(r'[.;!?]\s+(?=[A-ZÇĞİÖŞÜ(])', text[:match.start()])]
            start = starts[-1] if starts else 0
            sentence = text[start:match.end()]
            head = re.sub(r'^\(\d+\)\s*|^[a-zçğıöşü]\)\s*', '', sentence)
            comma = head.find(',')
            subject = head[:comma].strip() if 0 < comma <= 80 else head.split(' ')[0].strip()
            if not subject or subject not in text[:match.start()]:
                if soft:
                    continue
                return empty('INSUFFICIENT_EVIDENCE')
            rest = head[len(subject):].lstrip(', ').strip()
            modal = match.group()
            markers = len(list(LEGACY_MODAL.finditer(sentence)))
            if soft:
                markers += sum(1 for m in matches if not binding(m) and start <= m.start() < match.end())
            if markers > 1 or not rest.endswith(modal):
                if soft:
                    continue
                return empty('INSUFFICIENT_EVIDENCE')
            # "bildirmek zorundadır": the marker is a separate word and the action stops before
            # it; "bildirilir": the verb itself carries the duty and stays in the action.
            action = rest[:-len(modal)].rstrip() if standalone(modal) else rest
            if not action:
                if soft:
                    continue
                return empty('INSUFFICIENT_EVIDENCE')
            mode = modality(match.group())
            output.append({'source_quote': text, 'subject': subject, 'modality': mode,
                'required_action': None if mode.endswith('_NOT') else action,
                'prohibited_action': action if mode.endswith('_NOT') else None,
                'conditions': own(qualifier_tails(text, CONDITIONS)), 'exceptions': own(qualifier_tails(text, EXCEPTIONS)),
                'confidence_score': '0.5000'})
        if not output:
            return empty('INSUFFICIENT_EVIDENCE')
        return json.dumps({'status': 'EXTRACTED', 'obligations': output})
