import matplotlib.pyplot as plt
import numpy as np

def generate_color_palette(rows, cols):
    # 미리 정의된 기본 색상과 추가 색상을 포함한 팔레트 리스트 생성
    palette = [
        [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 165, 0), (0, 0, 0), (255, 255, 255)],
        [(255, 192, 203), (255, 0, 255), (0, 255, 255), (128, 0, 128), (128, 128, 0), (0, 128, 0), (0, 128, 128)],
        # 더 많은 색상 추가 가능
    ]

    # 추가 색상을 생성하고 팔레트 리스트에 추가
    for i in range(rows - len(palette)):
        row = []
        for j in range(cols):
            # 예시로 랜덤한 색상 생성
            r = (i * 255) // (rows - 1)
            g = (j * 255) // (cols - 1)
            b = 128
            row.append((r, g, b))
        palette.append(row)

    return palette

def plot_color_palette(palette):
    rows = len(palette)
    cols = len(palette[0])
    
    image = np.zeros((rows, cols, 3), dtype=np.uint8)
    
    for i in range(rows):
        for j in range(cols):
            r, g, b = palette[i][j]
            image[i, j, 0] = r
            image[i, j, 1] = g
            image[i, j, 2] = b
    
    plt.imshow(image)
    plt.axis('off')
    plt.show()

# 6x7 색상 팔레트 생성
colors = generate_color_palette(6, 7)

# 색상 팔레트 시각화
plot_color_palette(colors)