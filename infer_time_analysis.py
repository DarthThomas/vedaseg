import argparse
import statistics
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description='Infer time analysis')
    parser.add_argument('--raw', help='train config file path',
                        default='resnet18-orig-detail')
    args = parser.parse_args()
    return args


def basic_filter(lines):
    result = []
    for line in lines:
        if ':' in line and '%' not in line:
            result.append(line.strip())
    return result


def decode_line(line):
    split_res = line.split(':')
    if len(split_res) != 2:
        return None
    name, cost = split_res
    return name.strip(), float(cost.strip())


def main():
    args = parse_args()
    time_summery = defaultdict(list)
    time_summert_processed = defaultdict(dict)

    with open(args.raw, 'r') as f:
        lines = f.readlines()

    lines = basic_filter(lines)

    for line in lines:
        line_res = decode_line(line)
        if line_res is not None:
            time_summery[line_res[0]].append(line_res[1])

    print(f"\nIn {args.raw}:")
    print('-' * 50)
    for name, costs in time_summery.items():
        std = statistics.pstdev(costs[10:])
        avg = statistics.mean(costs[10:])
        time_summert_processed[name]['avg'] = avg
        time_summert_processed[name]['std'] = std
        tab = ' ' * 8 if name[0].isupper() else ''
        if name[1].isupper():
            tab = tab[:-4]

        print(f"{tab}{name}: {avg:.4f} +- {std:.5f}")
    print('-' * 50)

    total = 0
    for name, costs in time_summert_processed.items():
        if name[1].isupper():
            prec = costs['avg'] / time_summert_processed['infer cost']['avg']
            print(f"{name}: {prec * 100:.3f}%")
            total += prec * 100
    print(f"TOTAL: {total}\n")


if __name__ == '__main__':
    main()
