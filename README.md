Surgical Tools Pose Estimation

File	Description
combined_viewer.py:	    Main tracker (YOLO + SIFT + optical flow + pose)
combined_viewer_v2.py:	V2 with fine-tuned YOLO, manual ROI (t key), excluded classes
train_yolo.py:	        Fine-tuning script for the Roboflow surgical-tools dataset
depth_frame.py:	        Original Orbbec depth reference
rgb_test.py:      	    Original OpenCV RGB reference
pipeline.py:          	Original SIFT pipeline reference
yolo26n-seg.pt:        	YOLO segmentation model weights
.gitignore:            	Excludes pyorbbecsdk/, runs/, Log/, caches
