(define (domain warehouse)
	(:requirements :typing)
	(:types robot pallette - bigobject
        	location shipment order saleitem)

  	(:predicates
    	(ships ?s - shipment ?o - order)
    	(orders ?o - order ?si - saleitem)
    	(unstarted ?s - shipment)
    	(started ?s - shipment)
    	(complete ?s - shipment)
    	(includes ?s - shipment ?si - saleitem)

    	(free ?r - robot)
    	(has ?r - robot ?p - pallette)

    	(packing-location ?l - location)
    	(packing-at ?s - shipment ?l - location)
    	(available ?l - location)
    	(connected ?l - location ?l - location)
    	(at ?bo - bigobject ?l - location)
    	(no-robot ?l - location)
    	(no-pallette ?l - location)

    	(contains ?p - pallette ?si - saleitem)
  )

   (:action startShipment
      :parameters (?s - shipment ?o - order ?l - location)
      :precondition (and (unstarted ?s) (not (complete ?s)) (ships ?s ?o) (available ?l) (packing-location ?l))
      :effect (and (started ?s) (packing-at ?s ?l) (not (unstarted ?s)) (not (available ?l)))
   )
    
    (:action robotMove
      :parameters (?r - robot ?l1 - location ?l2 - location)
      :precondition (and (not (no-robot ?l1)) (connected ?l1 ?l2) (at ?r ?l1) (no-robot ?l2))
      :effect (and (no-robot ?l1) (not (no-robot ?l2)) (at ?r ?l2) (not (at ?r ?l1)))
   )
   
   (:action robotMoveWithPallette
      :parameters (?r - robot ?l1 - location ?l2 - location ?p - pallette)
      :precondition (and (no-robot ?l2) (at ?r ?l1) (or (free ?r) (has ?r ?p)) (connected ?l1 ?l2) (not (no-robot ?l1)) (not (no-pallette ?l1)) (at ?p ?l1) (no-pallette ?l2))
      :effect (and (no-robot ?l1) (no-pallette ?l1) (not (no-robot ?l2)) (not (no-pallette ?l2)) (has ?r ?p) (not (at ?r ?l1)) (not (at ?p ?l1)) (at ?r ?l2) (at ?p ?l2))
   )
   
   (:action moveItemFromPalletteToShipment
      :parameters (?l1 - location ?sh - shipment ?si - saleitem ?p - pallette ?o - order)
      :precondition (and (ships ?sh ?o) (orders ?o ?si) (started ?sh) (not (complete ?sh)) (not (includes ?sh ?si)) (packing-at ?sh ?l1) (contains ?p ?si) (at ?p ?l1))
      :effect (and (includes ?sh ?si) (not (contains ?p ?si)))
   )
   
   (:action completeShipment
      :parameters (?l1 - location ?sh - shipment ?o - order)
      :precondition (and (started ?sh) (ships ?sh ?o) (packing-at ?sh ?l1) (not (complete ?sh)))
      :effect (and (complete ?sh) (not (packing-at ?sh ?l1)) (available ?l1))
   )
)
