PDDL+ version of HVAC on ENSHP planner
===

When we submit this scalable planning model to conferences, some reviewers pointed out
 that we need the comparison between our Tensorflow model with nonlinear planners such as
 ENSHP. Although we claimed our work is very scalable there were no evidence to convince
  others.

So, thanks for Buser Say!! We show the evidence here.  In our planning problems, there is a
 HVAC domain. The files listed in this folder were used to test PDDL+ and ENSHP planner on
 HVAC problem with only two rooms and 12 hours. However, this planner was not able to
 give any result in one hour limit.
 
Remember:  Our model can easily handle this control optimize problem for an entire 
 building with 60 rooms and 96 hours. The optimization time was 4 mins. 
 Well, you can do it on even bigger problem.   