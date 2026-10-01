"""Compact model inputs and backend-owned citation construction."""
from .grounding import turkish, turkish_polarity
from .schema import ModelOutput, ExtractionOutput, Candidate, SupportingEvidence

CONTRACT_VERSION = 'compact-action-v1'
EXAMPLE = {'status':'EXTRACTED','obligations':[{'subject':'A firm','modality':'MUST',
    'action':'retain records.','conditions':[],'exceptions':[],'evidence':[]}]}

def sources(context):
    return {str(index):item for index,item in enumerate(context.items,1)} if context else {}

def model_context(context):
    if context is None:
        return {'sources':[],'dependency_gaps':[]}
    return {'sources':[{'source_id':sid,'text':item['text'],'reason':item['reason'],
                       'required':item.get('required',item['reason']=='exact_reference'),
                       'label':item.get('printed_label')} for sid,item in sources(context).items()],
            'dependency_gaps':list(context.dependency_gaps)}

def heading_field(context) -> dict:
    """{'provision_heading': ...} for the extraction payload when the packet names the provision's heading.

    v0.19: a sub-paragraph read alone ("(3) ... derhal alır.") loses what its article is about; the
    heading ("Üçüncü tarafa güven") gives it back. The prompt says it is context only; verify still
    requires every field to be a substring of the paragraph, so a heading copied into a field is rejected.
    """
    heading = getattr(context, 'heading', '') if context is not None else ''
    return {'provision_heading': heading[:200]} if heading else {}

def materialize(raw: str,text: str,context=None) -> ExtractionOutput:
    output=ModelOutput.model_validate_json(raw)
    known=sources(context)
    candidates=[]
    for proposed in output.obligations:
        supporting=[]
        seen=set()
        for evidence in proposed.evidence:
            source=known.get(evidence.source_id)
            if source is None or evidence.quote not in source['text']:
                raise ValueError('SUPPORT_SPAN_INVALID: quote does not occur in context source_id '+
                    repr(evidence.source_id)+'. Use a supplied context id and its exact text. '+
                    'If the quote comes from the primary paragraph, remove that evidence entry; '+
                    'keep the primary condition/exception in its corresponding field.')
            key=(source['section_id'],evidence.role)
            if key not in seen:
                supporting.append(SupportingEvidence(section_id=key[0],role=key[1],quote=source['text']))
                seen.add(key)
        # Polarity is the backend's: in Turkish the marker the copied action sits against
        # decides it. Measured with qwen3:4b: "hiç kimseye açıklayamazlar" came back as MUST.
        polarity=(turkish_polarity(text,proposed.action) if turkish(text) else None) or proposed.modality
        negative=polarity.endswith('_NOT')
        candidates.append(Candidate(source_quote=text,subject=proposed.subject,modality=polarity,
            required_action=None if negative else proposed.action,
            prohibited_action=proposed.action if negative else None,
            conditions=proposed.conditions,exceptions=proposed.exceptions,
            supporting_evidence=supporting,confidence_score='0.5000'))
    return ExtractionOutput(status=output.status,obligations=candidates)
