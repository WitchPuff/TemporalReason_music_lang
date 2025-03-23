rm -rf data/midi_segs
rm data/midi_seg_dict.json data/midi_oct/split_dict.json
python data/make_music_dataset.py &&
python data/convert_oct.py &&
python utils/train.py 