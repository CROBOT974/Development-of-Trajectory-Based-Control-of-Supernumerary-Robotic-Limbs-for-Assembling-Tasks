# Development of Trajectory Based Control of Supernumerary Robotic Limbs for Assembling Tasks
In this project, a control strategy of Supernumerary Robot Limbs (SRLs) based on trajectory generating is developed for an assembling task.This method is autonomous and could achieve high efficiency without the prerequisite of demonstration data or complex physical models.
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


