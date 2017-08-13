(define 
    (problem instance1)
    (:domain hvac)
    (:objects r1 r2 -room k11 k12 k13 k14 k15 k16 k17 k18 k19 k110 k111 k112 k21 k22 k23 k24 k25 k26 k27 k28 k29 k210 k211 k212 -request)
    (:init
        (= (time_requested r1 k11) 1)
        (= (temp_requested r1 k11) 21.75)
        
        (= (time_requested r1 k12) 2)
        (= (temp_requested r1 k12) 21.75)

        (= (time_requested r1 k13) 3)
        (= (temp_requested r1 k13) 21.75)
        
        (= (time_requested r1 k14) 4)
        (= (temp_requested r1 k14) 21.75)
        
        (= (time_requested r1 k15) 5)
        (= (temp_requested r1 k15) 21.75)

        (= (time_requested r1 k16) 6)
        (= (temp_requested r1 k16) 21.75)

        (= (time_requested r1 k17) 7)
        (= (temp_requested r1 k17) 21.75)

        (= (time_requested r1 k18) 8)
        (= (temp_requested r1 k18) 21.75)

        (= (time_requested r1 k19) 9)
        (= (temp_requested r1 k19) 21.75)

        (= (time_requested r1 k110) 10)
        (= (temp_requested r1 k110) 21.75)

        (= (time_requested r1 k111) 11)
        (= (temp_requested r1 k111) 21.75)

        (= (time_requested r1 k112) 12)
        (= (temp_requested r1 k112) 21.75)

	(= (time_requested r2 k21) 1)
        (= (temp_requested r2 k21) 21.75)
        
        (= (time_requested r2 k22) 2)
        (= (temp_requested r2 k22) 21.75)

        (= (time_requested r2 k23) 3)
        (= (temp_requested r2 k23) 21.75)
        
        (= (time_requested r2 k24) 4)
        (= (temp_requested r2 k24) 21.75)
        
        (= (time_requested r2 k25) 5)
        (= (temp_requested r2 k25) 21.75)

        (= (time_requested r2 k26) 6)
        (= (temp_requested r2 k26) 21.75)

        (= (time_requested r2 k27) 7)
        (= (temp_requested r2 k27) 21.75)

        (= (time_requested r2 k28) 8)
        (= (temp_requested r2 k28) 21.75)

        (= (time_requested r2 k29) 9)
        (= (temp_requested r2 k29) 21.75)

        (= (time_requested r2 k210) 10)
        (= (temp_requested r2 k210) 21.75)

        (= (time_requested r2 k211) 11)
        (= (temp_requested r2 k211) 21.75)

        (= (time_requested r2 k212) 12)
        (= (temp_requested r2 k212) 21.75)


        (= (temp r1) 10)
        (= (air_flow r1) 0)

	(= (temp r2) 10)
        (= (air_flow r2) 0)



        (= (time) 0)
        (= (comfort) 1.75)
	(= (obj) 0)


    )
    ;; the goal encodes the horizon of control. 
    (:goal 
        (and  (satisfied r1 k11)
	      (satisfied r1 k12)
              (satisfied r1 k13)
              (satisfied r1 k14)
              (satisfied r1 k15)
              (satisfied r1 k16)
              (satisfied r1 k17)
              (satisfied r1 k18)
              (satisfied r1 k19)
              (satisfied r1 k110)
              (satisfied r1 k111)
              (satisfied r1 k112)
              (satisfied r2 k21)
	      (satisfied r2 k22)
              (satisfied r2 k23)
              (satisfied r2 k24)
              (satisfied r2 k25)
              (satisfied r2 k26)
              (satisfied r2 k27)
              (satisfied r2 k28)
              (satisfied r2 k29)
              (satisfied r2 k210)
              (satisfied r2 k211)
              (satisfied r2 k212)
       )
    )
    (:metric minimize (obj))
)
