import numpy as np 
from PIL import Image
from scipy.ndimage import convolve
from scipy.signal import convolve2d
from dataclasses import dataclass
import streamlit as st

def resize_image(img):
	return img.resize((500,500))


def show_image(images):
	for img in images:
		try:
			img = Image.fromarray(img)
		except TypeError as e:
			print(f'Error reading image.\n{e}')
		finally:
			img.show()

def get_rgb(img: Image):
	try:
		img = np.array(img)
	except Excepion as e:
		raise e

	empty = np.zeros(img.shape[:-1])

	R = np.uint8(np.stack([img[:,:,0], empty, empty], axis=-1))
	G = np.uint8(np.stack([empty, img[:,:,1], empty], axis=-1))
	B = np.uint8(np.stack([empty, empty, img[:,:,2]], axis=-1))

	return (R, G, B)

def gaussian_kernel(size, sigma=1):
    kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-size//2)**2 + (y-size//2)**2)/(2*sigma**2)), (size, size))
    return kernel / np.sum(kernel)



@dataclass
class kernel:
	identity = np.identity(3)

	ridge_detection = np.array([0, -1, 0, -1, 4, -1, 0, -1, 0]).reshape(3, 3)

	vertical_edge = np.array([-1, 0, 1, -2, 0, 2, -1, 0, 1]).reshape(3,3)
	horizontal_edge = vertical_edge.T

	sharpen  = np.array([0, -1, 0, -1, 5, -1, 0, -1, 0]).reshape(3, 3)

	box_blur = np.ones((3, 3))/9
	box_blur_kernel_7x7 = np.ones((7, 7)) / 49
	gaussian_blur = (1/16)*np.array([1, 2, 1, 2, 4, 2, 1, 2, 1]).reshape(3,3)
	gaussian_kernel_5x5 = gaussian_kernel(5, sigma=1)

	emboss_kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
	
	laplacian_kernel = np.array([[0, 1, 0],[1, -4, 1],[0, 1, 0]])
	high_pass_kernel = np.array([[-1, -1, -1],[-1, 8, -1],[-1, -1, -1]])

	gradient_kernel_horizontal = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
	gradient_kernel_vertical = np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])

	diagonal_edge_kernel_tl_br = np.array([[-1, -1, 2],[-1, 2, -1],[2, -1, -1]])
	diagonal_edge_kernel_bl_tr = np.array([[2, -1, -1],[-1, 2, -1],[-1, -1, 2]])

	unsharp_masking_kernel = np.array([[-1, -1, -1],[-1,  9, -1],[-1, -1, -1]]) / 1

def get_kernel(kernel_name):
	if kernel_name == "ridge detection":
		return kernel.ridge_detection
	if kernel_name == 'vertical edge detection':
		return kernel.vertical_edge
	if kernel_name == 'horizontal edge detection':
		return kernel.horizontal_edge


if __name__ == '__main__':
	st.title("Imagething")
	selected_kernel = st.selectbox(
		'Kernel:',
		("ridge detection", "vertical edge detection", "horizontal edge detection"),
		index = None,
		placeholder = 'select a kernel ...'
		)

	#st.subheader("upload image:")
	uploaded_image = st.file_uploader('Upload image:', type=['png', 'jpg', 'jpeg'], label_visibility='visible')
	if uploaded_image is not None:
		img = Image.open(uploaded_image).convert('RGB')
		st.image(resize_image(img))

		img = np.array(img)

		k = get_kernel(selected_kernel)

		img[:,:,0] = np.uint8(np.clip(convolve2d(img[:,:,0], k, mode='same'), 0, 255))
		img[:,:,1] = np.uint8(np.clip(convolve2d(img[:,:,1], k, mode='same'), 0, 255))
		img[:,:,2] = np.uint8(np.clip(convolve2d(img[:,:,2], k, mode='same'), 0, 255))

		img = Image.fromarray(img)

		st.text("Result:")
		st.image(resize_image(img))
