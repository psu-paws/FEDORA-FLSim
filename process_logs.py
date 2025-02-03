from pathlib import Path
import re

def process_number(number_str: str) -> int:
    return int(number_str) if "tensor(" not in number_str  else int(number_str .split("[")[1].split("]")[0])

def process_log(log_path: Path):
    for line in log_path.open("r"):
        if "Namespace(" in line:
            args = line.split("(")[1].split(")")[0].split(", ")
            args = {x.split("=")[0]: x.split("=")[1] for x in args}
            total_reads = 0
            wasted_reads = 0
            lost_reads = 0
            lost_samples = 0
            total_samples = 0
        elif "Total reads" in line:
            m = re.match(r"Total requests (?P<total_req>([0-9]+)), Total reads (?P<total_r>([0-9]+|tensor\(\[[0-9]+\]\))), Wasted reads (?P<wasted_r>([0-9]+|tensor\(\[[0-9]+\]\))), Lost reads (?P<lost_r>([0-9]+|tensor\(\[[0-9]+\]\))), Total samples (?P<total_samples>([0-9]+|tensor\(\[[0-9]+\]\))), Max # feat per user (?P<max_feature_per_user>([0-9]+|tensor\(\[[0-9]+\]\)))", line)
            if m is None:
                print(line)
            total_reads += int(m.group('total_r')) if "tensor(" not in m.group("total_r") else int(m.group("total_r").split("[")[1].split("]")[0])
            wasted_reads += int(m.group('wasted_r')) if "tensor(" not in m.group("wasted_r") else int(m.group("wasted_r").split("[")[1].split("]")[0])
            lost_reads += int(m.group('lost_r')) if "tensor(" not in m.group("lost_r") else int(m.group("lost_r").split("[")[1].split("]")[0])
            total_samples += process_number(m.group('total_req'))
            lost_samples += process_number(m.group('lost_r'))
        elif "Total test AUC" in line:
            auc = float(line.split(" ")[-1])
        elif "FL training done" in line:
            # print(args)
            # print(total_reads, wasted_reads, lost_reads, total_samples, lost_samples)
            # print(auc)
            
            return auc, total_reads, wasted_reads, lost_reads, total_samples, lost_samples
    return None

def sort_key(line):
    
    mode_map = {
        "pub": 0,
        "hide_priv_val": 1,
        "hide_number_of_priv_val": 2
    }
    
    dataset_map = {
        "taobao": 0,
        "movielens": 1,
    }
    
    mode = line[0]
    mode_key = mode_map[mode]
    
    dataset = line[1]
    dataset_key = dataset_map[dataset]
    
    eps = line[2]
    if eps is not None:
        eps = -eps
        
    return (mode_key, dataset_key, eps)

def main():
    
    output_list = []
    for path in Path().glob('*.log'):
        print(f"Processing {path}")
        components = path.stem.split("-")
        
        if len(components) < 2:
            continue
        
        dataset = components[0]
        
        mode = components[1]
        
        if len(components) >= 3:
            eps = float(components[2].split("_")[1])
            display_name = f"{dataset}-{mode}-eps_{eps}"
        else:
            eps = None
            display_name = f"{dataset}-{mode}"

        print(f"Config identified as {display_name}")
        
        result = process_log(path)
        
        if result is None:
            print(f"{display_name} is incomplete, skipping")
            continue
        
        auc, total_reads, wasted_reads, lost_reads, total_samples, lost_samples = result
        
        print(auc, total_reads, wasted_reads, lost_reads, total_samples, lost_samples)
        
        if mode == "pub":
            reduced_accesses = None
            lost_read_percent = None
            dummy_reads_percent = None
        else:
            reduced_accesses = (1 - total_reads / total_samples) * 100
            lost_read_percent = lost_reads / total_reads * 100
            dummy_reads_percent = wasted_reads / total_reads * 100
        
        print(f"{reduced_accesses=}")
        print(f"{lost_read_percent=}")
        print(f"{dummy_reads_percent=}")
        
        output_list.append((
            mode, dataset, 
            eps, 
            reduced_accesses, 
            dummy_reads_percent, 
            lost_read_percent, 
            auc
        ))
    
    output_list.sort(key=sort_key)
    
    with open("results.csv", "w") as outfile:
        outfile.write("mode, dataset, eps, reduced_accesses, dummy, lost, auc\n")
        
        for line in output_list:
            
            mode, dataset, eps, reduced_accesses, dummy_reads_percent, lost_read_percent, auc = line
            
            outfile.write(", ".join(map(str, (
            mode, dataset, 
            f"{eps}" if eps is not None else "-", 
            f"{reduced_accesses:.2f}%" if reduced_accesses is not None else "-", 
            f"{dummy_reads_percent:.2f}%" if dummy_reads_percent is not None else "-", 
            f"{lost_read_percent:.2f}%" if lost_read_percent is not None else "-", 
            f"{auc:.4f}"
            ))) + "\n")
        
        

if __name__ == "__main__":
    main()
