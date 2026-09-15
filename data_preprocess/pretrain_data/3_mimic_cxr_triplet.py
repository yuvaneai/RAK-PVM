import os
import pandas as pd
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from radgraph import RadGraph

rg = RadGraph()

def parse_radgraph_output(data):

    def get_modified_text(eid):
        entity = data[eid]
        modifiers = []

        for k, v in data.items():
            if ['modify', eid] in v['relations']:
                modifiers.append(v['tokens'])

        if modifiers:
            return " ".join([m.lower() for m in modifiers] + [entity['tokens'].lower()])

        return entity['tokens'].lower()

    triples = []
    for eid, entity in data.items():

        if not entity['label'].startswith('Observation'):
            continue

        exist = entity['label'].split("::")[-1].lower()
        entity_name = get_modified_text(eid)

        position = None
        for rel in entity['relations']:
            if rel[0] == 'located_at':
                loc_id = rel[1]
                position = get_modified_text(loc_id)

        position = "none" if position is None else position.lower()

        triples.append((entity_name.lower(), position, exist))

    df = pd.DataFrame(triples, columns=["entity", "position", "existence"])
    df = df.sort_values(by="entity").reset_index(drop=True)
    return df

def filter_entities(df):

    entities = df["entity"].tolist()
    unique_entities = list(set(entities))

    to_remove = set()
    for e1 in unique_entities:
        for e2 in unique_entities:
            if e1 == e2:
                continue
            if e1 in e2:
                to_remove.add(e1)

    final_entities = [e for e in unique_entities if e not in to_remove]

    df_filtered = df[df["entity"].isin(final_entities)].copy()
    df_filtered = df_filtered.sort_values(by="entity").reset_index(drop=True)

    return df_filtered

def collapse_to_dict_cell(df):

    result_dict = {}

    for i, row in df.reset_index(drop=True).iterrows():
        idx = str(i + 1)
        result_dict[idx] = {
            "entity": row["entity"],
            "position": row["position"],
            "existence": row["existence"]
        }

    return str(result_dict)

def process_dataframe(df):
    triplet_list = []

    for idx, row in df.iterrows():
        report = row["report"]

        if idx % 100 == 0:
            print(f"Processing row {idx}/{len(df)} ...")

        if not isinstance(report, str) or report.strip() == "":
            triplet_list.append("{}")
            continue

        try:
            predictions = rg(report)
            entities = predictions["0"]["entities"]

            df_parsed = parse_radgraph_output(entities)
            df_filtered = filter_entities(df_parsed)
            triplet_cell = collapse_to_dict_cell(df_filtered)

        except Exception as e:
            triplet_cell = "{}"

        triplet_list.append(triplet_cell)

    df["Triplet Extraction"] = triplet_list
    return df

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(BASE_DIR, "./result/2_master_select.csv")
output_file = os.path.join(BASE_DIR, "./result/3_mimiccxr_master_triplet.csv")

df = pd.read_csv(input_file)
df_processed = process_dataframe(df)
df_processed.to_csv(output_file, index=False)

print("Done:", output_file)