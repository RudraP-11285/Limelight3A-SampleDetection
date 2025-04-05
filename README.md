# Limelight3A-SampleDetection
Using contour detection on a Limelight3A camera to single out samples in the submersible, FTC Into the Deep (2024-2025)

The Challenge:
Detect samples from the submersible using contour detection to filter out yellow, blue, and red samples and return position and rotation values for our robot to use to autonomously grab samples accurately.


The Algorithm:
Use CV2 and numpy to calculate contour areas and simplify them into boxes, ensuring that the extra small contours are deleted in order to only return data for a single contour (of each color). Using the returned angle and position relative to the camera, we are able to move our robot around to line up with the block.

You can get the Limelight Output Data in Android Studio using:

<img width="449" alt="Screenshot 2025-04-05 at 11 02 41 AM" src="https://github.com/user-attachments/assets/2d9f6f5f-4b6d-415e-b18c-e8c0949121a5" />




Using CV2 we can simplify the image using contours and create bounding boxes around the samples. The information is then stored in lists and sent to the Control Hub, where it can be accessed through the Java code on the robot.
