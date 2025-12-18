from pathlib import Path
from scripts.generate_report import main as gen_main


def test_report_creates_file(tmp_path, monkeypatch):
    out_dir = tmp_path
    # run the generator with small parameters and project lead set to 'testlead'
    monkeypatch.setattr('sys.argv', ['scripts/generate_report.py', '--repeats', '2', '--train-episodes', '10', '--eval-hands', '50', '--seed', '0', '--lead', 'testlead'])
    gen_main()
    rpt = Path('reports/summary.html')
    assert rpt.exists()
    assert rpt.stat().st_size > 0
