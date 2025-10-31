# pip install vcdvcd
from vcdvcd import VCDVCD
import argparse, statistics as stats

def average_low_between_fall_and_rise(vcd_path, signal_path, t0=None, t1=None):
    vcd = VCDVCD(vcd_path, only_sigs=[signal_path], store_tvs=True)
    tv = vcd[signal_path]['tv']  # [(time, '0'/'1'/...)]
    # VCD timescale 그대로 사용: 필요하면 t0, t1도 같은 단위로 지정

    # 정규화: value를 0/1로만 취급 (x/z는 무시하거나 직전값 유지)
    norm = []
    last = None
    for t, v in tv:
        v = v.decode() if isinstance(v, bytes) else v
        if v in ('0', '1'):
            last = int(v)
            norm.append((t, last))
        elif last is not None:
            norm.append((t, last))
    if not norm:
        return None, {'count':0}

    # 클립 범위 설정
    if t0 is None: t0 = norm[0][0]
    if t1 is None: t1 = norm[-1][0]

    # 구간 시작 시 값 파악 (필요 시 보간)
    # norm에는 변화점만 있으므로, t0 시점의 값을 알아낸다.
    cur_val = norm[0][1]
    for t, v in norm:
        if t > t0:
            break
        cur_val = v

    low_durs = []
    in_low = (cur_val == 0)
    t_fall = t0 if in_low else None

    # 스캔
    for t, v in norm:
        if t < t0:  # 구간 전은 무시
            continue
        if t > t1:  # 구간 끝을 넘으면 종료
            break

        # 엣지 감지
        if not in_low and v == 0:        # 1 -> 0 : falling
            in_low = True
            t_fall = t
        elif in_low and v == 1:          # 0 -> 1 : rising (구간 하나 종료)
            t_rise = t
            # 클리핑하여 유효 부분만 계수
            start = max(t_fall, t0)
            end   = min(t_rise, t1)
            if end > start:
                low_durs.append(end - start)
            in_low = False
            t_fall = None

    # 구간 끝에서 아직 LOW라면 T1까지 반영
    if in_low and t_fall is not None:
        start = max(t_fall, t0)
        end   = t1
        if end > start:
            low_durs.append(end - start)

    if not low_durs:
        return None, {'count':0}

    avg = sum(low_durs) / len(low_durs)
    meta = {
        'count': len(low_durs),
        'min': min(low_durs),
        'max': max(low_durs),
        'median': stats.median(low_durs),
        'timescale': vcd.timescale,  # 예: ('1', 'ns')
    }
    return avg, meta

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vcd", required=True)
    ap.add_argument("--signal", required=True, help="예: top.dut.sig")
    ap.add_argument("--t0", type=int, default=None, help="구간 시작 (VCD timescale 단위)")
    ap.add_argument("--t1", type=int, default=None, help="구간 끝 (VCD timescale 단위)")
    args = ap.parse_args()
    avg, meta = average_low_between_fall_and_rise(args.vcd, args.signal, args.t0, args.t1)
    if meta['count'] == 0:
        print("해당 구간에 LOW 구간( fall→rise )이 없습니다.")
    else:
        ts = meta['timescale']
        print(f"count={meta['count']}, avg={avg} {ts[0]}{ts[1]}, "
              f"min={meta['min']}, max={meta['max']}, median={meta['median']}")