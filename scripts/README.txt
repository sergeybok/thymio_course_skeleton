Submission of Sergiy's and Timone's MyT assignment

TO RUN:

terminal 1:
$ roslaunch thymio_course_skeleton thymio_gazebo_bringup.launch name:=thymio10 world:=wall

terminal 2:
$ rosrun thymio_course_skeleton task{number}.py thymio10




Task comments:

task 1 (draw 8): Implemented it with 2 circles. The robot finishes one
circle and starts the other, as a function of time. Meaning we calculated
the time it takes to finish a circle with the formula
T = pi*diameter / velocity


task 2 (orthogonal to wall): We wanted to use law of cosine and sine to
estimate current angle difference between the wall and thymio however the
front center sensor wasn't working well on the Gazebo. So instead we opted
for the simpler algorithm of turning until the center-right and center-left
front sensors are equal, and if they are then that means the thymio is
orthogonal to the wall.


task 3 (back to wall and move away): We weanted to use back sensors for
this to similarly compare left and right back sensor and stop when they
are equal, however our back sensors don't work in Gazebo (see discussion
in forum). So what we did was reuse our code from task 2, and then do a
180 degree turn when our condition of being orthogonal to the wall (task2)
is fulfilled, meaning we will be facing directly away from the wall after.
Then we drive away from wall as a function of time
for 2 meters


task 4 (8 in real life): works, video should be attached


task 5 (orthogonal to wall in real life): works, video should be attached




