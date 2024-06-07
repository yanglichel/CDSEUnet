import numpy as np
import os  # 用于遍历文件夹
import nibabel as nib  # 用nibabel包打开nii文件
import imageio  # 图像io


def nii_to_image(niifile):
    filenames = os.listdir(filepath)  # 指定nii所在的文件夹
    for f in filenames:
        # 开始读取nii文件
        img_path = os.path.join(filepath, f)
        img = nib.load(img_path, )  # 读取nii
        img_fdata = img.get_fdata()

        fnamex = f.replace('.nii.gz', ' -x')  # 去掉nii的后缀名创建x方向2D图像文件夹
        img_f_pathx = os.path.join(imgfile, fnamex)  # 创建nii对应的x方向2D图像的文件夹
        if not os.path.exists(img_f_pathx):
            os.mkdir(img_f_pathx)  # 新建文件夹

        fnamey = f.replace('.nii.gz', ' -y')  # 去掉nii的后缀名创建y方向2D图像文件夹
        img_f_pathy = os.path.join(imgfile, fnamey)  # 创建nii对应的y方向2D图像的文件夹
        if not os.path.exists(img_f_pathy):
            os.mkdir(img_f_pathy)  # 新建文件夹

        fnamez = f.replace('.nii.gz', ' -z')  # 去掉nii的后缀名创建z方向2D图像文件夹
        img_f_pathz = os.path.join(imgfile, fnamez)  # 创建nii对应的z方向2D图像图像的文件夹
        if not os.path.exists(img_f_pathz):
            os.mkdir(img_f_pathz)  # 新建文件夹

        # 开始转换为图像
        # 可能用到的图像变换
        # 旋转操作利用numpy 的rot90（a,b）函数 a为数据 b为旋转90度的多少倍 ！正数逆时针 负数顺时针
        # 左右翻转 ： img_lr = np.fliplr(img) 上下翻转： img_ud = np.flipud(img)
        (x, y, z) = img.shape  # 获取图像的3个方向的维度
        for i in range(x):  # x方向
            silce = img_fdata[i, :, :]  # 选择哪个方向的切片都可以 不要忘了i改到对应方向
            imageio.imwrite(os.path.join(img_f_pathx, '{}.png'.format(i)), silce)  # 保存图像
        for i in range(y):  # y方向
            silce = np.rot90(img_fdata[:, i, :], 1)
            imageio.imwrite(os.path.join(img_f_pathy, '{}.png'.format(i)), silce)  # 保存图像
        for i in range(z):  # z方向
            silce = np.fliplr(np.rot90(img_fdata[:, :, i], -1))
            imageio.imwrite(os.path.join(img_f_pathz, '{}.png'.format(i)), silce)  # 保存图像


if __name__ == '__main__':
    filepath = r'D:\PycharmProjects\yjs_data\yjs2\nii'
    imgfile =r'D:\PycharmProjects\yjs_data\yjs2\png'
    nii_to_image(filepath)

