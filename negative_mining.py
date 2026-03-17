import json

def extract_icd_tokens(data_dict, format_as_special_token=False):
    """
    Recursively extracts all ICD codes from a nested dictionary.
    
    Args:
        data_dict (dict): The dictionary containing the ICD data.
        format_as_special_token (bool): If True, formats 'A00' as '<ICD_A00>'.
    """
    all_codes = []

    def _traverse(current_level):
        for code, details in current_level.items():
            if format_as_special_token:
                all_codes.append(f"<ICD_{code}>")
            else:
                all_codes.append(code)
            
            sub_diagnoses = details.get("sub_diagnoses", {})
            if sub_diagnoses:
                _traverse(sub_diagnoses)

    if "diagnoses" in data_dict:
        _traverse(data_dict["diagnoses"])
    else:
        _traverse(data_dict)

    return all_codes


def build_hard_negative_lookup(data_dict, format_as_special_token=True):
    """
    Builds a dictionary mapping every ICD code to a list of its siblings (Hard Negatives).
    Returns: { "<ICD_A00.0>": ["<ICD_A00.1>", "<ICD_A00.9>"], ... }
    """
    hard_negatives_map = {}

    def _format(code):
        return f"<ICD_{code}>" if format_as_special_token else code

    def _traverse(current_level):
        siblings = list(current_level.keys())
        
        for code, details in current_level.items():
            formatted_code = _format(code)
            
            negatives = [_format(sib) for sib in siblings if sib != code]
            hard_negatives_map[formatted_code] = negatives
            
            sub_diagnoses = details.get("sub_diagnoses", {})
            if sub_diagnoses:
                _traverse(sub_diagnoses)

    if "diagnoses" in data_dict:
        _traverse(data_dict["diagnoses"])
    else:
        _traverse(data_dict)

    return hard_negatives_map


if __name__ == "__main__":
    with open('icd10cm.json', 'r') as f:
        icd_data = json.load(f)

    raw_codes = extract_icd_tokens(icd_data, format_as_special_token=False)
    icd_special_tokens = extract_icd_tokens(icd_data, format_as_special_token=True)
    print(f"Extracted {len(icd_special_tokens)} total codes!")

    negatives_lookup = build_hard_negative_lookup(icd_data, format_as_special_token=True)
    print("Example lookup for <ICD_A01.01>:", negatives_lookup.get("<ICD_A01.01>", []))