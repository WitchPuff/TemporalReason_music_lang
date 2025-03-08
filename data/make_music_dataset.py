import os
import random
import pretty_midi
import zipfile
from tqdm import tqdm
import string
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
BASE62_ALPHABET = string.ascii_letters + string.digits

def base62_encode(num):
    """Base62编码"""
    if num == 0:
        return BASE62_ALPHABET[0]
    base62 = []
    while num:
        num, rem = divmod(num, 62)
        base62.append(BASE62_ALPHABET[rem])
    return ''.join(reversed(base62))

def base62_decode(encoded):
    """Base62解码"""
    num = 0
    for char in encoded:
        num = num * 62 + BASE62_ALPHABET.index(char)
    return num

def encode_path_base62(path):
    """路径转换为 Base62 编码"""
    return base62_encode(int.from_bytes(path.encode(), 'big'))

def decode_path_base62(encoded):
    """Base62 解码回原路径"""
    num = base62_decode(encoded)
    return num.to_bytes((num.bit_length() + 7) // 8, 'big').decode()





def extract_midi_segment(midi_data, start, duration, output_path):
    new_midi = pretty_midi.PrettyMIDI()
    
    for instrument in midi_data.instruments:
        new_instrument = pretty_midi.Instrument(program=instrument.program, 
                                                is_drum=instrument.is_drum, 
                                                name=instrument.name)
        for note in instrument.notes:
            if round(note.start, 6) >= round(start, 6) and round(note.end, 6) <= round(start + duration, 6):
                new_note = pretty_midi.Note(velocity=note.velocity,
                                            pitch=note.pitch,
                                            start=round(note.start - start, 6), 
                                            end=round(note.end - start, 6))
                new_instrument.notes.append(new_note)
        if new_instrument.notes:
            new_midi.instruments.append(new_instrument)
    
    new_midi.write(output_path)

def check(relation, t1, d1, t2, d2, total_duration):
    if t1 + d1 > total_duration or t2 + d2 > total_duration:
        return False
    if relation == "before":
        return t1 < t1 + d1 < t2 < t2 + d2
    elif relation == "meets":
        return t1 < t1 + d1 == t2 < t2 + d2
    elif relation == "overlaps":
        return t1 < t2 < t1 + d1 < t2 + d2
    elif relation == "starts":
        return t1 == t2 < t2 + d2 < t1 + d1
    elif relation == "during":
        return t1 < t2 < t2 + d2 < t1 + d1
    elif relation == "finishes":
        return t1 < t2 < t2 + d2 == t1 + d1
    elif relation == "equals":
        return t1 == t2 < t2 + d2 == t1 + d1
    else:
        raise ValueError("未知的关系类型！")

def generate_relation_samples(midi_path, relations = ["before", "meets", "overlaps", "starts", "during", "finishes", "equals"], num_pairs_per_relation=2, output_dir='data/midi_segs'):
    path_hash = encode_path_base62(midi_path)
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    total_duration = round(midi_data.get_end_time(), 6)
    
    if total_duration < 31:
        raise ValueError("MIDI文件时长不足30秒！")
    

    os.makedirs(output_dir, exist_ok=True)
    
    sample_pairs = {rel: [] for rel in relations}
    
    for rel in relations:
        for pair_idx in range(num_pairs_per_relation):
            max_attempts = 100
            for attempt in range(max_attempts):
                d1 = round(random.uniform(15, min(30, total_duration / 2)), 6)  # 第一段的时长
                d2 = round(random.uniform(15, min(30, total_duration / 2)), 6)  
                if rel not in ["before", "meets", "overlaps"]: 
                    # make sure d1 >= d2
                    if abs(d1 - d2) < 1:
                        continue
                    dt = d1
                    d1 = max(d2, dt)
                    d2 = min(d2, dt)
                t1 = round(random.uniform(0, total_duration - d2 - d1 - 0.1), 6)
                if rel == "before":
                    # t1 < t1+d1 < t2 < t2+d2
                    t2 = round(random.uniform(t1 + d1 + 0.5, total_duration - d2 - 0.1), 6)

                elif rel == "meets":
                    # t1 < t1+d1 = t2 < t2+d2
                    t2 = round(t1 + d1, 6)
                
                elif rel == "overlaps":
                    # t1 < t2 < t1+d1 < t2+d2
                    lower = round(t1 + 0.1, 6)
                    upper = round(t1 + d1 - 0.1, 6)
                    t2 = round(random.uniform(lower, upper), 6)
                
                elif rel == "starts":
                    # t1 = t2 < t2+d2 < t1+d1, d1 > d2
                    t2 = t1
                
                elif rel == "during":
                    # t1 < t2 < t2+d2 < t1+d1, d1 > d2
                    lower = round(t1 + 0.1, 6)
                    upper = round(t1 + d1 - d2 - 0.1, 6)
                    t2 = round(random.uniform(lower, upper), 6)
                
                elif rel == "finishes":
                    # t1 < t2 < t2+d2 = t1+d1, d1 > d2
                    t2 = round(t1 + d1 - d2, 6)
                
                elif rel == "equals":
                    # t1 = t2 < t2+d2 = t1+d1, d1 = d2
                    d2 = d1
                    t2 = t1
                
                else:
                    raise ValueError("未知的关系类型！")

                if check(rel, t1, d1, t2, d2, total_duration):
                    break  
                else:
                    continue  
            
            seg1_filename = "{}_pair{}_seg1_{}.mid".format(rel, pair_idx, path_hash)
            seg2_filename = "{}_pair{}_seg2_{}.mid".format(rel, pair_idx, path_hash)
            seg1_path = os.path.join(output_dir, seg1_filename)
            seg2_path = os.path.join(output_dir, seg2_filename)
            
            extract_midi_segment(midi_data, t1, d1, seg1_path)
            extract_midi_segment(midi_data, t2, d2, seg2_path)
            
            sample_pairs[rel].append((seg1_path, seg2_path))
    
    return {path_hash: sample_pairs}





# 假设 generate_relation_samples 是 CPU 密集型任务
def process_midi_file(midi_file):
    try:
        samples_dict = generate_relation_samples(midi_file)
        return midi_file, samples_dict
    except Exception as e:
        return midi_file, None

if __name__ == '__main__':
    # 读取 ZIP 文件
    data_zip = zipfile.ZipFile('data/midi.zip', 'r')
    file_list = ['data/' + '/'.join(n.split('/')[-3:]).replace('._', '') 
                 for n in data_zip.namelist() 
                 if n.lower().endswith(('.mid', '.midi'))]
    
    print('Number of midi tracks:', len(file_list))

    seg_dict = {}
    failed = []
    relations = ["before", "meets", "overlaps", "starts", "during", "finishes", "equals"]
    count = {rel: 0 for rel in relations}

    # 设置多进程池，使用 CPU 核心数量
    num_workers = os.cpu_count() or 4  # 默认 4 线程
    print("Number of workers: ", num_workers)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_file = {executor.submit(process_midi_file, midi_file): midi_file for midi_file in file_list}

        for future in tqdm(as_completed(future_to_file), total=len(file_list), desc="Processing MIDI files"):
            midi_file, samples_dict = future.result()
            if samples_dict is None:
                failed.append(midi_file)
                continue

            seg_dict.update(samples_dict)
            for pairs in samples_dict.values():
                for rel, segs in pairs.items():
                    count[rel] += len(segs)

    print('Number of samples:', count)
    print('Number of failed:', len(failed), failed)

    # 存储结果
    with open('data/midi_seg_dict.json', 'w') as f:
        json.dump(seg_dict, f, indent=4)