"""Small synthetic regression benchmark, not an expert-labelled FCA legal evaluation."""
import argparse
import json
from collections import Counter

from regchain.evidence import digest
from .pipeline import extract
from .providers import configured_provider
from .service import PROMPT_HASH

# Gold labels are explicit, not produced by the extractor under test.
CASES = [
    {'id':'duty','text':'A firm must retain records.','gold':[['A firm','MUST','retain records.']]},
    {'id':'prohibition','text':'A firm must not delete records.','gold':[['A firm','MUST_NOT','delete records.']]},
    {'id':'guidance','text':'A firm should review records.','gold':[['A firm','SHOULD','review records.']]},
    {'id':'permission','text':'A firm may retain copies.','gold':[['A firm','MAY','retain copies.']]},
    {'id':'exception','text':'A firm must retain records unless exempt.','gold':[['A firm','MUST','retain records unless exempt.']]},
    {'id':'condition','text':'A firm must notify customers if fees change.','gold':[['A firm','MUST','notify customers if fees change.']]},
    {'id':'two_duties','text':'A firm must retain records. A firm must review controls.',
     'gold':[['A firm','MUST','retain records.'],['A firm','MUST','review controls.']]},
    {'id':'external_context','text':'A firm must act in accordance with CONC 7.3.','gold':[],
     'abstention_expected':True},
    {'id':'informational','text':'This publication describes a consultation.','gold':[]},
]


def evaluate(provider) -> dict:
    tp = fp = fn = supported_quotes = predictions = 0
    details = []
    for case in CASES:
        result = extract(case['text'],provider)
        actual = Counter((o.subject,o.modality,o.required_action or o.prohibited_action) for o in result.output.obligations)
        expected = Counter(tuple(row) for row in case['gold'])
        tp += sum((actual & expected).values())
        fp += sum((actual - expected).values())
        fn += sum((expected - actual).values())
        predictions += len(result.output.obligations)
        supported_quotes += sum(o.source_quote == case['text'] for o in result.output.obligations)
        details.append({'id':case['id'],'status':result.output.status,'reason':result.reason,
                        'candidate_count':len(result.output.obligations)})
    ratio = lambda n,d: n/d if d else None
    return {'dataset':'synthetic-extraction-v1','dataset_hash':digest(CASES),'sample_count':len(CASES),
            'provider':provider.name,'model_version':provider.model_version,'prompt_hash':PROMPT_HASH,
            'metrics':{'exact_extraction_precision':ratio(tp,tp+fp),'exact_extraction_recall':ratio(tp,tp+fn),
                       'false_negative_rate':ratio(fn,tp+fn),'verbatim_citation_accuracy':ratio(supported_quotes,predictions)},
            'counts':{'tp':tp,'fp':fp,'fn':fn},'cases':details,
            'limitations':'Synthetic exact-span regression only; no legal-accuracy, applicability, gap or semantic hallucination claim.'}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--provider',choices=['rules','ollama'],required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(configured_provider(args.provider)),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
