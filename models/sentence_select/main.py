
#!/usr/bin/env python3
import argparse
import models.sentence_select.model_zoo as model_zoo
from models.sentence_select.utils import Vocabulary

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='Path to the data of in-hospital mortality task',
                        default='/data/joe/physician_notes/mimic-data/preprocessed/')
    parser.add_argument('--feature_period', type=str, help='feature period',
                        choices=["24", "48", "retro"])
    parser.add_argument('--feature_used', type=str, help='feature used', default="notes",
                        choices=["all", "notes", "all_but_notes"])
    parser.add_argument('--note', type=str, help='feature used',
                        choices=["physician", "physician_nursing", "discharge", "all", "all_but_discharge"])
    parser.add_argument('--task', type=str, help='task',
                        choices=["mortality", "readmission"])
    parser.add_argument('--split', type=str, help='task',
                        choices=["train", "valid", "test"])
    parser.add_argument('--segment', type=str, help='segmentation type')
    parser.add_argument('--filter', type=str, help='segmentation type', default=None)
    parser.add_argument('--reverse_label', action="store_true", default=False)
    parser.add_argument('--model', type=str, help='task',
                        choices=["LR", "DAN", "GRUD"])
    parser.add_argument('--device', type=str, help='task',
                        default="0")
    args = parser.parse_args()
    print (args)

    import pathlib
    path = f"{args.data}/select_sentence/similarity/"
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    if args.model == "LR":
        model = model_zoo.LR(args)
    if args.model == "DAN":
        model = model_zoo.DeepAverageNetwork(args)
    model.predict()
