

def generate_sequence_file(n,file_name):
    with open(file_name, 'w') as file:
        for i in range(n):
            file.write(f"{i}\n")
            left = i - 1
            right = i + 1
            neighbors = []
            while len(neighbors) < 2:
                if left >= 0 and left not in neighbors:
                    neighbors.append(left)
                    left -= 1
                if right < n and right not in neighbors:
                    neighbors.append(right)
                    right += 1
                if left < 0 and right >= n:
                    break
            neighbors.sort()
            decrement_num = 2000
            neighbor_strs = []
            for neighbor in neighbors:
                neighbor_strs.append(f"{neighbor} {decrement_num}")
                decrement_num = max(1000, decrement_num - 100)
            line = f"{len(neighbors)} {' '.join(neighbor_strs)}\n"
            file.write(line)


try:
    
    generate_sequence_file(49,'/home/zhaoyibin/3DRE/3DGS/FatesGS/DTU/diff/scan37/pair.txt')
    print("文件已生成，名为 output.txt")
except ValueError:
    print("输入无效，请输入一个有效的整数。")
    