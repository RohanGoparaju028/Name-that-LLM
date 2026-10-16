"""Regenerate the self-contained Kaggle notebook from train.py defaults."""
import inspect
import json
from pathlib import Path

from train import run


def main():
    defaults = inspect.signature(run).parameters
    settings = ', '.join(
        f'{name}={defaults[name].default!r}'
        for name in ['epochs', 'hidden', 'components', 'seed']
    )
    source = Path('train.py').read_text().split('def main():')[0]
    execution = (
        'with threadpool_limits(limits=2):\n'
        '    submission = run(\n'
        '        data_dir=Path("/kaggle/input/competitions/name-that-llm"),\n'
        '        output_dir=Path("/kaggle/working"),\n'
        f'        {settings},\n'
        '    )\n'
        'submission.head()\n'
    )
    cells = [{
        'cell_type': 'markdown', 'metadata': {},
        'source': ['# Name That LLM: single hidden layer\n',
                   'Run all cells with Internet disabled. Validation holds out entire prompts; '
                   'the final pipeline is refitted on all labeled passages.\n'],
    }]
    for code in [source, execution]:
        compile(code, '<notebook>', 'exec')
        cells.append({
            'cell_type': 'code', 'metadata': {}, 'execution_count': None,
            'outputs': [], 'source': code.splitlines(keepends=True),
        })
    for index, cell in enumerate(cells):
        cell['id'] = f'cell-{index}'
    notebook = {
        'cells': cells, 'nbformat': 4, 'nbformat_minor': 5,
        'metadata': {
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python', 'version': '3'},
        },
    }
    Path('notebooks/name_that_llm.ipynb').write_text(json.dumps(notebook, indent=2))


if __name__ == '__main__':
    main()
