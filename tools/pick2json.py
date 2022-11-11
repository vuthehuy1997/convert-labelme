import os
import json

CUR_DIR = os.getcwd()

if __name__ == '__main__':
    OUTPUT_DIR = 'outputs_cavet_pred_s2s_pick_v3'
    txt_filelist = os.listdir(os.path.join(CUR_DIR, OUTPUT_DIR))
    for txt_filename in txt_filelist:
        data = {}
        txt_lines = open(os.path.join(OUTPUT_DIR, txt_filename), 'r').readlines()
        for line in txt_lines:
            if line[-1] == '\n':
                line = line[:-1]
            column_name, text = line.split('\t')
            text = text.replace('_', ' ')
            print(column_name, text)
            data[column_name] = text
        json_filename = txt_filename[:-3] + 'json'
        with open(os.path.join(CUR_DIR, 'outputs_pred_s2s_pick_v3', json_filename), 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
