;;Planning for controlling hvac (Heating, Ventilation and  Air-Conditioning). First very coarse abstraction, which uses a discrete interepretation of the problem where #t = 1 
;;and ignores many thermal aspects of the problem
(define 
    (domain hvac)
    (:types room request -object)
    (:predicates
        (satisfied ?l -room ?r -request)
    )
    (:functions 
        (air-flow ?l -room) ;; air-flow rate 
        (temp ?l -room) ;;temperature of the room. This variable is not controlled directly
        (time) ;; this keeps track of the time. This is meant to allow us to capture the specific time-slot
        (time_requested ?l -room ?r -request) ;; this captures the time slot of interest. Depending on the request there could be many of them
        (temp_requested ?l -room ?r -request) ;; this models the particular temperature request
        (comfort) ;; this is a constant regulating the actual difference between the desired and the perceived temperature
	(obj)
    )

    
;;    (:action satisfier
;;        :parameters (?l -room ?r -request)
;;        :precondition (and (<= (temp ?l) (+ (temp_requested ?l ?r) (comfort))) (>= (temp ?l) (- (temp_requested ?l ?r) (comfort))) (= (time) (time_requested ?l ?r)))
;;        :effect (and (assign (air_flow ?l) 0.0) (satisfied ?l ?r))
;;    )

     (:action metric_positive1
        :parameters (?l -room ?r -request)
        :precondition (and (>= (temp ?l) (temp_requested ?l ?r) ) (= (time) (time_requested ?l ?r)))
        :effect (and (assign (air_flow ?l) 0.0) (satisfied ?l ?r) (increase (obj) (* 20000 (- (temp ?l) (temp_requested ?l ?r) ) ) ) )
     )

     (:action metric_negative1
        :parameters (?l -room ?r -request)
        :precondition (and (<= (temp ?l) (temp_requested ?l ?r) ) (= (time) (time_requested ?l ?r)))
     :effect (and (assign (air_flow ?l) 0.0) (satisfied ?l ?r) (increase (obj) (* 20000 (- (temp_requested ?l ?r) (temp ?l) ) ) ) )
     )

     (:action metric_positive2
        :parameters (?l -room ?r -request)
        :precondition (and (>= (comfort) (- (temp ?l) (temp_requested ?l ?r))) (<= 0 (- (temp ?l) (temp_requested ?l ?r))) (= (time) (time_requested ?l ?r)))
        :effect (and (assign (air_flow ?l) 0.0) (satisfied ?l ?r) (increase (obj) (* 10 (- (temp ?l) (temp_requested ?l ?r) ) ) ) )
    )

    (:action metric_negative2
        :parameters (?l -room ?r -request)
        :precondition (and (>= (comfort) (- (temp_requested ?l ?r) (temp ?l))) (<= 0 (- (temp_requested ?l ?r) (temp ?l))) (= (time) (time_requested ?l ?r)))
     :effect (and (assign (air_flow ?l) 0.0) (satisfied ?l ?r) (increase (obj) (* 10 (- (temp_requested ?l ?r) (temp ?l) ) ) ) )
  )
  
    
    ;; this process models the passing of time
    (:process time_passing
       :parameters ()
       :precondition ()
       :effect (increase  (time) (* #t 1)) 
    )

    ;; this process models how the temperature changes along the time according to the air-flow and the temp_sa. Many other parameters have to be added. 
    ;; At this stage the temperature of the room uniquely depends on the enthalpy
    (:process thermal_change
       :parameters (?l ?l2 -room)
       :precondition ()
       :effect (and (increase  (temp ?l) 
       (* (/ #t 80) (+ (/ (- (temp ?l2) (temp ?l) ) 1.5) (+ (/ (- 6 (temp ?l) ) 4) (+ (/ (- 10 (temp ?l) ) 2) (* (air_flow ?l) (* 1.006 (- 40 (temp ?l) ) ) ) ) ) ) ) ) )
    )

    ;; the next actions model the intervention on the air_flow and on the supply air temperature
    (:action increase_air_flow
       :parameters (?l -room)
       :precondition (and (<= (air_flow ?l) 9) )
       :effect (and(increase (air_flow ?l) 1) )
    )
        
) 
