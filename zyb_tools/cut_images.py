from PIL import Image


def cut_image(image_path):
    try:
        # 打开图片
        image = Image.open(image_path)
        width, height = image.size

        # 确定中间方块的大小
        size = min(width, height)
        left = (width - size) // 2
        top = (height - size) // 2
        right = (width + size) // 2
        bottom = (height + size) // 2

        # 裁剪中间方块
        middle_square = image.crop((left, top, right, bottom))

        # 计算每一块的大小
        piece_width = size // 4
        piece_height = size // 4

        # 切割成16块并保存
        for i in range(4):
            for j in range(4):
                left = j * piece_width
                top = i * piece_height
                right = (j + 1) * piece_width
                bottom = (i + 1) * piece_height
                piece = middle_square.crop((left, top, right, bottom))
                piece.save(f"/home/zhaoyibin/windows_tongbu/文章/第三篇文章/0048_{i}_{j}.png")
        print("图片切割并保存成功！")
    except Exception as e:
        print(f"发生错误: {e}")


if __name__ == "__main__":
    image_path = "/home/zhaoyibin/windows_tongbu/文章/第三篇文章/0048.png"  # 替换为你的图片路径
    cut_image(image_path)
    