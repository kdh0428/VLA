"""Clean pdfLaTeX/BibTeX build with atomic publication of the two verified PDFs."""
from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent
PAPER = ROOT / 'paper'
EXPECTED_STYLE = {
    'cvpr.sty': '1cd7f434b13d41c475df4a1901a5d82d9ca8b80b8cf6205dc4602a5c164286b4',
    'ieeenat_fullname.bst': 'e38e6166bd7b1e6d23a1b79dcdb55c656e4fcdbe91bdf6b50d827e6b5d1aacfc',
}
for name, digest in EXPECTED_STYLE.items():
    assert hashlib.sha256((PAPER / name).read_bytes()).hexdigest() == digest
for executable in ('pdflatex', 'bibtex'):
    if not shutil.which(executable): raise SystemExit(executable + ' is required')
qa = {'engine': 'isolated clean pdfLaTeX + BibTeX', 'new_gpu_inference': False,
      'matches_official_CVPR2027_v1_template_hashes': True, 'documents': {}}
with tempfile.TemporaryDirectory(prefix='cvpr-v04-clean-') as tmp:
    work = Path(tmp)
    for source in PAPER.iterdir():
        if source.suffix in ('.tex', '.bib', '.bst', '.sty'):
            shutil.copy2(source, work / source.name)
    shutil.copytree(PAPER / 'figures', work / 'figures')
    for name in ('main', 'supplementary'):
        commands = [['pdflatex', '-interaction=nonstopmode', '-halt-on-error', '-file-line-error', name + '.tex'],
                    ['bibtex', name]]
        commands += [commands[0]] * 3
        console = []
        for cmd in commands:
            run = subprocess.run(cmd, cwd=work, text=True, capture_output=True, timeout=180)
            console.append(run.stdout + run.stderr)
            if run.returncode:
                (ROOT / 'qa' / (name + '_FAILED_BUILD_v04.txt')).write_text('\n'.join(console))
                raise SystemExit((run.stdout + run.stderr)[-5000:])
        pdf = work / (name + '.pdf')
        content = pdf.read_bytes()
        assert content.rstrip().endswith(b'%%EOF'), name
        doc = PdfReader(pdf)
        texts = [p.extract_text() or '' for p in doc.pages]
        aux = (work / (name + '.aux')).read_text(errors='replace')
        reference_label = re.search(r'\\newlabel\{page:references\}\{\{[^}]*\}\{(\d+)\}', aux)
        assert reference_label, (name, 'missing page:references label')
        references = [int(reference_label.group(1))]
        log = (work / (name + '.log')).read_text(errors='replace')
        errors = bool(re.search(r'^!|LaTeX Error|Fatal error', log, flags=re.M))
        undefined = bool(re.search(r'Citation .+ undefined|Reference .+ undefined|There were undefined', log))
        overfull = log.count('Overfull')
        entry = {'pages': len(doc.pages), 'references_pages': references,
                 'latex_errors': errors, 'undefined_refs_or_citations': undefined,
                 'overfull_boxes': overfull, 'sha256': hashlib.sha256(content).hexdigest()}
        assert not errors and not undefined and overfull == 0, (name, entry)
        if name == 'main':
            assert references == [9] and len(doc.pages) == 10, entry
            llm_label = re.search(r'\\newlabel\{page:llm-usage-statement\}\{\{[^}]*\}\{(\d+)\}', aux)
            assert llm_label and int(llm_label.group(1)) == 9
            assert 'LLM Usage Statement' in re.sub(r'\s+', ' ', texts[8])
            entry['llm_usage_statement_page'] = 9
        else:
            assert references and references[0] >= 18 and len(doc.pages) >= references[0], entry
            joined = re.sub(r'\s+', ' ', '\n'.join(texts))
            assert 'pose-group sensitivity' in joined
            assert 'Additional Decoding and Motion Visualizations' in joined
        qa['documents'][name] = entry
        print(json.dumps({'validated': name, **entry}), flush=True)
        (work / (name + '_BUILD_CONSOLE_20261023_v04.txt')).write_text('\n'.join(console))
    # Publish only after both final PDFs validate.
    for name, export in [('main', 'CVPR_PAPER_20261023_v04.pdf'),
                         ('supplementary', 'CVPR_SUPPLEMENT_20261023_v04.pdf')]:
        for suffix in ('.pdf', '.aux', '.log', '.bbl', '_BUILD_CONSOLE_20261023_v04.txt'):
            source = work / (name + suffix)
            target = PAPER / source.name
            staged = target.with_name(target.name + '.tmp')
            staged.write_bytes(source.read_bytes())
            os.replace(staged, target)
        target = ROOT / export
        staged = target.with_name(target.name + '.tmp')
        staged.write_bytes((work / (name + '.pdf')).read_bytes())
        os.replace(staged, target)
(ROOT / 'qa/BUILD_QA_20261023_v04.json').write_text(json.dumps(qa, indent=2) + '\n')
print(json.dumps(qa, indent=2))
