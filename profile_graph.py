import re
import matplotlib.pyplot as plt
import random

result_list = []
pattern = re.compile(r'^\d+,\s*\d+,\s*\d+,\s*\d+$')


# file open
def read_log(src):
    with open(src, "r") as f:
        for line in f:
            line = line.strip()
            if pattern.match(line):
                nums = [int(x) for x in line.split(",")]
                result_list.append(nums)
    return result_list


# random color
def generate_random_color():
    # Generate a random color in hex format
    random_color = "#{:06x}".format(random.randint(0, 0xFFFFFE))
    return random_color


# visualize
def visualize(list):
        
        time_ranges = {'mvin': [], 'mvout': [], 'compute': []}
        tags = {'mvin': [], 'mvout': [], 'compute': []}
        labels = {'mvin': [], 'mvout': [], 'compute': []}
        colors = {'mvin': [], 'mvout': [], 'compute': []}
        visual_data = {}
        
        for i in range(len(list)):
            visual_data[i] = {'Gemmini': list[i][0],
                                'Type': list[i][1], 
                                'Latency': (list[i][2], list[i][3]-list[i][2]),
                                'Color': generate_random_color()
                                }


        gemmini_num = [visual_data[tag]['Gemmini'] for tag in visual_data]
        max_gemmini_num = max(gemmini_num) + 1

        starts = [visual_data[tag]['Latency'][0] for tag in visual_data]
        durations = [visual_data[tag]['Latency'][1] for tag in visual_data]
        ends = [visual_data[tag]['Latency'][0] + visual_data[tag]['Latency'][1] for tag in visual_data]

        fig, axes = plt.subplots(max_gemmini_num, 1, figsize=(12, 4*max_gemmini_num), sharex=True)

        if max_gemmini_num == 1:
            axes = [axes]
        
        for i in range(max_gemmini_num):
            ax = axes[i]

            ax.axvspan(min(starts), max(ends), 60, facecolor='lightgrey', alpha=0.3)

            for tag in visual_data:
                if (visual_data[tag]['Gemmini'] == i):
                    if(visual_data[tag]['Type']==0):
                        ax.broken_barh([visual_data[tag]['Latency']], (20, 8), facecolors=visual_data[tag]['Color'], edgecolor='none')
                        #ax.text(visual_data[tag]['Latency'][0]+visual_data[tag]['Latency'][1] / 2, 24, tag, ha='center', va='center', color='black', fontsize=10)
                    if(visual_data[tag]['Type']==1):
                        ax.broken_barh([visual_data[tag]['Latency']], (10, 8), facecolors=visual_data[tag]['Color'], edgecolor='none')
                        #ax.text(visual_data[tag]['Latency'][0]+visual_data[tag]['Latency'][1] / 2, 14, tag, ha='center', va='center', color='black', fontsize=10)
                    if(visual_data[tag]['Type']==2):
                        ax.broken_barh([visual_data[tag]['Latency']], (0, 8), facecolors=visual_data[tag]['Color'], edgecolor=visual_data[tag]['Color'])            
                        #ax.text(visual_data[tag]['Latency'][0]+visual_data[tag]['Latency'][1] / 2, 4, tag, ha='center', va='center', color='black', fontsize=10)

            ax.set_xlim(min(starts)-(max(ends) - min(starts))/20, max(ends)+(max(ends) - min(starts))/20)
            ax.set_xlabel("Time")
            ax.set_yticks([24, 14, 4])
            ax.set_yticklabels(["ld", "ex", "st"])
            ax.tick_params(axis='y', which='both', length=0)
            ax.set_title(f"Gemmini_{i}")

        plt.tight_layout(h_pad=4.0)
        print("Overall Start Time:", min(starts))
        print("Overall End Time:", max(ends))
        print("duration:", max(ends) - min(starts))