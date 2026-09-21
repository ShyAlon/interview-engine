from pathlib import Path

def test_production_engine_has_no_console_input_or_require_event():
 text='\n'.join(p.read_text() for p in Path('src/interview_engine').glob('*.py'))
 assert 'input(' not in text
 assert 'REQUIRE_EVENT' not in text
 assert '[y/N]' not in text
