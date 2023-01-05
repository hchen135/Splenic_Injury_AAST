import cv2
from skimage.io import imread,imsave
import numpy as np

def find_ori_image_info(GT_content,image_id):
	images = GT_content['images']
	for image in images:
		if image['id'] == image_id:
			return image
	return None

def find_annotations(GT_content,image_id):
	annotations = []
	for anno in GT_content['annotations']:
		if anno['image_id'] == image_id:
			annotations.append(anno)
	return annotations

def draw_boxes(image_prediction,class_name,color=[255,0,0]):
	image = cv2.imread(image_prediction['file_name'])
	if image.shape[2] == 1:
		image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
	boxes = image_prediction['boxes']
	scores = image_prediction['scores']

	if len(boxes) == 0:
		return image

	for i in np.argsort(scores):
		x1, y1, x2, y2 = boxes[i]
		score = scores[i]
		print(image.shape)
		cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)

		text_str = '%s: %.2f' % (class_name, score)

		font_face = cv2.FONT_HERSHEY_DUPLEX
		font_scale = 0.6
		font_thickness = 1

		text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

		text_pt = (int(x1), int(y1 - 3))
		text_color = [255, 255, 255]

		cv2.rectangle(image, (int(x1), int(y1)), (int(x1 + text_w), int(y1 - text_h - 4)), color, -1)
		cv2.putText(image, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)

	return image

def draw_masks(coco,image_id,color=[0,255,0]):
	cat_ids = coco.getCatIds()
	anns_ids = coco.getAnnIds(imgIds=image_id, catIds=cat_ids, iscrowd=None)
	anns = coco.loadAnns(anns_ids)

	mask = coco.annToMask(anns[0])
	for i in range(len(anns)):
		mask += coco.annToMask(anns[i])
	mask = mask > 0

	image = imread(coco.imgs[image_id]['file_name'])
	if len(image.shape) == 2:
		image = np.stack([image,image,image],axis=2)
	image[mask] = color
	return image
	




