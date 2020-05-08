# cv2TermProject
Alden and Xing's CV2 term project for Dr. Adam Czajka's class at ND.

### Xing's yolo transfer learning experiments

For yolo transfer learning go into folder darknet_new. The requirements are the same as those in practical 7. To see the popup of the tagged pictures on crc gpu machines:
1. Log into crc via ssh username@crcfe02.crc.nd.edu.
2. Open and interactive session via qsh -q gpu -l gpu=1.
3. Set gpu to an available gpu via setenv CUDA_VISIBLE_DEVICES $SGE_HGR_gpu_card
4. Load cuda and opencv vi module load cuda/10.0 opencv
5. Unzip the folder as long as your running on gpu this should work.

From our preliminary results, we found that our raw tagged object data was not enough for the system to learn the correct tagging of clustered objects. To run an example of this run the following:
head data/datasets/valid_obj_green.txt | ./darknet cfg/yolov3_objects_only.cfg backup/yolov3_objects_only_final.weights

![Figure 1. Training on objects only](/report_images/objects_only.png)

We also combined the individual object pictures with the tagged bin pictures the results can be seen by run the following:
head data/valid_totes.txt | ./darknet cfg/yolov3_set_aside.cfg backup/yolov3_set_aside_2000.weights
We see that this models is able to tag some of the pictures but is proned to tag the whole bin as frisbee. This lead us to believe that the size of the individual images were too big. Since the neural network is used to seeing the Fribee take up the entire frame it is prone to tag the entire frame as on big frisbee.


![Figure 2. Training on all images](/report_images/set_aside.png)

To address this we did the following experiments:
1. As a baseline, we looked at a random sample of tagged bin data and how well the system performed if we only used tagged bin data with a random 80/20 split. We found that this gave up the best performance with an IOU of 68.8% overall. 
  a. Run the following to see examples: head data/valid_totes.txt | ./darknet cfg/yolov3_totes.cfg backup/yolov3_totes_final.weights
  b. For the average IOU run python get_iou.py results/totes/!
![Figure 3. Training on all bin images](/report_images/totes.png)
  
2. We resized the images of the individual objects and added padding around it so that it would be proportionally similar to what we expect to see in the bin photos. We tried padding it with white pixels similar to the background and combined it with our bin data. Again we do a random 80/20 split and we get an average IOU of 24.5%
  a. Run the following to see examples: head data/valid_white.txt | ./darknet cfg/yolov3_resize_white.cfg backup/yolov3_resize_white_final.weights
  b. For the average IOU run python get_iou.py results/resize_white/
![Figure 4. Training on all objects resized with white pixels and bin images](/report_images/resize_white_3.png) 

3. We resized those same images but pad with green pixels since our objects are in the green plastic tote. Doing the same thing as in experiment 2 we get an average IOU of 47.42% The improvement here could be due to overfitting so we use both data in further experiments.
  a. Run the following to see examples: head data/valid_2.txt | ./darknet cfg/yolov3_resize.cfg backup/yolov3_resize_final.weights
  b. For the average IOU run python get_iou.py results/resize_green/
![Figure 5. Training on all objects resized with green pixels and bin images](/report_images/resize_green_3.png)   
To further understand what the neural network is capable of learning from what it can't we divide the bin data based on features of the lighting, angle, and sensor. This is so we can see what has a greater effect on the models abiliy to learn.

4. Ligthing: for this experiment we include resized images along with all bin images taken under al1 lighting conditions. The validation set is thus all the images taken under the other lighting conditions. This we found was challenging for the system to learn because there was only 1/3 of all bin data in training and two different lighting conditions to test for. 
a. head data/datasets/valid_al1.txt | ./darknet cfg/al1_lighting.cfg backup/al1_lighting_final.weights for resized images with white pixel paddings. To get IOU run python get_iou.py results/al1/ we get 4.6%
b. head data/datasets/valid_al1_green.txt | ./darknet cfg/al1_lighting.cfg backup/al1_lighting_final.weights for resized images with green pixel paddings. To get IOU run python get_iou.py results/al1_green/ we get 8.5%

5. Angle: for this experiment all top view pictures were added and side view pictures became validation data. We found that this was the hardest condition to learn for the system.
a. head data/datasets/valid_top.txt | ./darknet cfg/top_view_only.cfg backup/top_view_only_final.weights for resized images with white pixel paddings. To get IOU run python get_iou.py results/top_view/ we get 3.8%
a. head data/datasets/valid_top_green.txt | ./darknet cfg/top_view_green.cfg backup/top_view_green_final.weights for resized images with green pixel paddings. To get IOU run python get_iou.py results/top_green/ we get 4.8%

6. High Resolution Senors: for this experiment all pictures taken by our mobi were added and all the webcam pictures became validation data. We found that this was the easiest condition to learn for the system.
a. head data/datasets/valid_mobi.txt | ./darknet cfg/mobi.cfg backup/mobi_final.weights for resized images with white pixel paddings. To get IOU run python get_iou.py results/mobi/ we get 13.47%
a. head data/datasets/valid_mobi_green.txt | ./darknet cfg/mobi_green.cfg backup/mobi_green_final.weights for resized images with green pixel paddings. To get IOU run python get_iou.py results/mobi_green/ we get 13.97%

7. High Resolution Senors: for this experiment all pictures taken by our mobi were added and all the webcam pictures became validation data. We found that this was the easiest condition to learn for the system.
a. head data/datasets/valid_c615.txt | ./darknet cfg/c615.cfg backup/c615_final.weights for resized images with white pixel paddings. To get IOU run python get_iou.py results/c615/ we get 5.45%
a. head data/datasets/valid_c615_green.txt | ./darknet cfg/c615_green.cfg backup/c615_green_final.weights for resized images with green pixel paddings. To get IOU run python get_iou.py results/c615_green/ we get 7.84%

The results of test can be seen by running head data/test.txt in any of the above situations in place of the command for looking at validation data. For almost all of these systems none gave us false positives. The exceptions are experiment 3 which mistook part of a blue bin for the frisbee.

From these experiments it seem that individual objects alone are not enough for the system to recognize it in the wild. We definitely need some number of in context data. With that it is also very important scale the individual objects such that they aren't disportionally larger than what we expect to see in context. Lastly certain changes are easy for the system to compensate for. For example, going from higher resolution to lower resolution is much easier than the other way around, and varying angles are very hard for the system to recognize. 
