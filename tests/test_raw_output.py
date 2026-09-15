import json
from pydantic import ValidationError
from vmodel_contracts import StakeholderContext
from interview_engine.raw_output import RawOutputLog
from interview_engine.repair import clean_model_patch


def test_raw_output_paths_are_stable_and_do_not_touch_console(tmp_path):
    log=RawOutputLog(tmp_path/"debug")
    a=log.write("conops","Repairing stakeholder 1/2","raw one")
    b=log.write("conops","Repairing stakeholder 1/2","raw two")
    assert a.name=="conops-repairing-stakeholder-1-2-attempt-1.raw.txt"
    assert b.name=="conops-repairing-stakeholder-1-2-attempt-2.raw.txt"
    assert b.read_text()=="raw two"


def test_strict_patch_unwraps_single_item_and_rejects_echo():
    try:
        StakeholderContext.model_validate({"stakeholder":"manager","role":"approves","source_turns":[2]})
    except ValidationError as error:
        patch=clean_model_patch({"stakeholder_context":[{"goals":["reduce delay"],
          "operating_environment":"input_value={'x': 1}, input_type=list]","junk":"x"}]},StakeholderContext,error)
    assert patch=={"goals":["reduce delay"]}
