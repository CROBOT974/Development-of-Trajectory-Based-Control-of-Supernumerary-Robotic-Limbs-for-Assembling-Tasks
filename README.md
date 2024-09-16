# Development of Trajectory-Based Control of Supernumerary Robotic Limbs for Assembling Tasks
In this project, a control strategy of Supernumerary Robot Limbs (SRLs) based on trajectory generating is developed for an assembling task.This method is autonomous and could achieve high efficiency without the prerequisite of demonstration data or complex physical models. It utilized a improved RRT-Connect method for trajectory planning, and tracing the hand to accomplish human-robot coorperation.

## Install
Here is the code installing testing in Windows 11.

* **Downloading the codes**
```
git clone https://github.com/CROBOT974/Development-of-Trajectory-Based-Control-of-Supernumerary-Robotic-Limbs-for-Assembling-Tasks.git
```
* **Installing the required packages**

The inverse kinematics function is relied on [ikpy](https://github.com/Phylliade/ikpy), and the [fcl-python](https://github.com/BerkeleyAutomation/python-fcl/tree/master) library is referred for coolision detection. 
```
pip install numpy
pip install scipy
pip install mujoco-viewer
pip install ikpy
pip install python-fcl
```
## Run a demo
* **Test the whole senario**
```
python -m tests.main
```
* **Check the Graph**

graph1: Comparing the trajectory of between the human hand and robot gripper.
graph2: Comparing the desired and real joint angles.
```
python -m tests.graph.graph1
python -m tests.graph.graph2
```
## Result

* **demo of the scenario**

[Demo](https://youtube.com/shorts/IFuw9-X2uf0?feature=share)

* **Improved RRT-Connect (Compared to original RRT-Connect)**

original RRT-Connect
![Figure arm2](https://github.com/user-attachments/assets/6ed6fd0e-cb83-4843-b888-f402ee83af7e)

improved RRT-Connect
![Figure arm](https://github.com/user-attachments/assets/38d92fcc-a4ad-48d4-b6fb-85135a5a790f)


* **Trajectory Tracing in Cartesain Space**
![result2_2](https://github.com/user-attachments/assets/2a1de80d-a933-4b99-aba5-d6e8af96fff1)

* **Joint Angles**
![result3_1](https://github.com/user-attachments/assets/3a4da374-c752-4a05-8f25-b7b0b96e8b04)
![result3_2](https://github.com/user-attachments/assets/48a5910c-fb24-42af-aced-451fd3249a71)
![result3_3](https://github.com/user-attachments/assets/5011fc3b-86db-4e74-9ee9-01db9b3fcabd)
![result3_4](https://github.com/user-attachments/assets/0f8bbb84-97f8-4a60-9bb4-a16f1d6747b6)
![result3_5](https://github.com/user-attachments/assets/673914f8-f5f5-4a19-8504-7a8079dc7d76)
![result3_6](https://github.com/user-attachments/assets/7e2359c5-98ab-4c18-9d66-8b8c94c12682)
![result3_7](https://github.com/user-attachments/assets/40610326-0e10-46cb-84aa-061bc7b3ad37)



