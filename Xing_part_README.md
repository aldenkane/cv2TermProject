# cv2TermProject
Alden and Xing's CV2 term project for Dr. Adam Czajka's class at ND.
## (B) Methods and Models

### Xing's yolo transfer learning experiments

For yolo transfer learning go into folder darknet_new. The requirements are the same as those in practical 7. To see the popup of the tagged pictures on crc gpu machines:
1. Log into crc via ssh username@crcfe02.crc.nd.edu.
2. Open and interactive session via qsh -q gpu -l gpu=1.
3. Set gpu to an available gpu via setenv CUDA_VISIBLE_DEVICES $SGE_HGR_gpu_card
4. Load cuda and opencv vi module load cuda/10.0 opencv
5. unzip the folder

From our preliminary results, we found that our raw tagged object data was not enough for the system to learn the correct tagging of clustered objects. To run an example of this run the following:
head data/datasets/valid_obj_green.txt | ./darknet cfg/yolov3_objects_only.cfg backup/yolov3_objects_only_final.weights

To address this we did the following experiments:
1. As a baseline, we looked at a random sample of tagged bin data and how well the system performed if we only used tagged bin data with a random 80/20 split. We found that this gave up the best performance with an IOU of 68.8% overall. 
a. Run the following to see examples: head data/valid_totes.txt | ./darknet cfg/yolov3_totes.cfg backup/yolov3_totes_final.weights
b. For the average IOU run python get_iou.py results/totes/
2. We resized the images of the individual objects and added padding around it so that it would be proportionally similar to what we expect to see in the bin photos. We tried padding it with white pixels similar to the background and combined it with our bin data. Again we do a random 80/20 split and we get an average IOU of 24.5%
3. We resized those same images but pad with green pixels since our objects are in the green plastic tote. Doing the same thing as in experiment 2 we get an average IOU of 47.42% The improvement here could be due to overfitting so we use both data in further experiments.
