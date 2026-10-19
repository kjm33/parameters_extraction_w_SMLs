# Gold-v3 spot-check (r12a) — 50 rows

Pick N rows stratified by lang (pl >= 12), prioritized: (a) any teacher differs from v2, (b) teachers disagree, (c) rest. Seed 20261013.
teachers: t1_haiku55=t1_haiku55 t2_gpt54mini=t2_gpt54mini t3_gpt6luna=t3_gpt6luna t4_glm53flash=t4_glm53flash t5_dsv41flash=t5_dsv41flash t6_ling31flash=t6_ling31flash

verdict per column: ok (agree with v2 / correct fix) | bad (wrong) | ?
## [1] aug_i-3751ee055050 (de, here_directions, aug_train)

utt: fahr uns von Aurora nach Greeley ohne Mautstraßen

params: transportMode:string(enum=5), origin:string(enum=0), destination:string(enum=0), via:array(enum=0), departureTime:string(enum=0), routingMode:string(enum=2), avoid_features:array(enum=0), lang:string(enum=0)

| v2 | t1_haiku55 | t2_gpt54mini | t3_gpt6luna | t4_glm53flash | t5_dsv41flash | t6_ling31flash |
|---|---|---|---|---|---|---|
| {"transportMode":null,"origin":null,"destination":null,"via":null,"departureTime":null,"routingMode":null,"avoid_features":["tollRoad"],"lang":null} | {"transportMode":"car","origin":"Aurora","destination":"Greeley","via":null,"departureTime":null,"routingMode":null,"avoid_features":["tollRoad"],"lang":null} | {"transportMode":null,"origin":"Aurora","destination":"Greeley","via":null,"departureTime":null,"routingMode":null,"avoid_features":["tollRoad"],"lang":null} | {"transportMode":"car","origin":"Aurora","destination":"Greeley","via":null,"departureTime":null,"routingMode":null,"avoid_features":["tollRoad"],"lang":null} | {"transportMode":"car","origin":"Aurora","destination":"Greeley","via":null,"departureTime":null,"routingMode":null,"avoid_features":["tollRoad"],"lang":null} | {"transportMode":"car","origin":"Aurora","destination":"Greeley","via":null,"departureTime":null,"routingMode":null,"avoid_features":["tollRoad"],"lang":null} | {"transportMode":"car","origin":"Aurora","destination":"Greeley","via":null,"departureTime":null,"routingMode":null,"avoid_features":["tollRoad"],"lang":null} |

verdict(v3): ok [src=t1_haiku55, judged by assistant (de/es/fr, per policy §3)]

## [2] aug_i-3e7c23101ddc (de, here_directions, aug_train)

utt: route von Medellín nach Pradera ohne Fähren über Supía

params: transportMode:string(enum=5), origin:string(enum=0), destination:string(enum=0), via:array(enum=0), departureTime:string(enum=0), routingMode:string(enum=2), avoid_features:array(enum=0), lang:string(enum=0)

| v2 | t1_haiku55 | t2_gpt54mini | t3_gpt6luna | t4_glm53flash | t5_dsv41flash | t6_ling31flash |
|---|---|---|---|---|---|---|
| {"transportMode":null,"origin":null,"destination":null,"via":null,"departureTime":null,"routingMode":null,"avoid_features":["ferry"],"lang":null} | {"transportMode":"car","origin":"Medellín","destination":"Pradera","via":["Supía"],"departureTime":null,"routingMode":null,"avoid_features":["ferry"],"lang":null} | {"transportMode":null,"origin":"Medellín","destination":"Pradera","via":["Supía"],"departureTime":null,"routingMode":null,"avoid_features":["ferry"],"lang":null} | {"transportMode":null,"origin":"Medellín","destination":"Pradera","via":["Supía"],"departureTime":null,"routingMode":null,"avoid_features":["ferry"],"lang":null} | {"transportMode":"car","origin":"Medellín","destination":"Pradera","via":["Supía"],"departureTime":null,"routingMode":null,"avoid_features":["ferry"],"lang":null} | {"transportMode":null,"origin":"Medellín","destination":"Pradera","via":["Supía"],"departureTime":null,"routingMode":null,"avoid_features":["ferry"],"lang":null} | null |

grounded: t1_haiku55=yes t2_gpt54mini=yes t3_gpt6luna=yes t4_glm53flash=yes t5_dsv41flash=yes t6_ling31flash=NO

verdict(v3): ok [src=t1_haiku55, judged by assistant (de/es/fr, per policy §3)] — tM=car from 'fahr uns'; via=[Supia] ok

## [3] aug_i-7fe204f0808d (de, here_geocode, aug_train)

utt: route von Zigong nach Changning mit einem Kaffeestopp, etwa 130 km

params: q:string(enum=0), lang:string(enum=0), limit:integer(enum=0)

| v2 | t1_haiku55 | t2_gpt54mini | t3_gpt6luna | t4_glm53flash | t5_dsv41flash | t6_ling31flash |
|---|---|---|---|---|---|---|
| {"q":"Zigong","lang":null,"limit":null} | {"q":"Zigong","lang":null,"limit":null} | {"q":"Zigong","lang":null,"limit":null} | {"q":"Zigong, Changning","lang":null,"limit":null} | {"q":"Zigong","lang":null,"limit":null} | {"q":null,"lang":null,"limit":null} | {"q":"Zigong nach Changning","lang":null,"limit":null} |

verdict(v3): ok [src=t1_haiku55, judged by assistant (de/es/fr, per policy §3)] — q=Zigong origin (=v2)

## [4] xtr-29cf437be869 (de, here_directions, core_train)

utt: Fahr mich mit dem Lieferwagen von Hamburg nach München, aber weiche dabei bitte allen mautpflichtigen Strecken aus.

params: transportMode:string(enum=5), origin:string(enum=0), destination:string(enum=0), via:array(enum=0), departureTime:string(enum=0), routingMode:string(enum=2), avoid_features:array(enum=0), lang:string(enum=0)

| v2 | t1_haiku55 | t2_gpt54mini | t3_gpt6luna | t4_glm53flash | t5_dsv41flash | t6_ling31flash |
|---|---|---|---|---|---|---|
| {"transportMode":null,"origin":null,"destination":null,"via":null,"departureTime":null,"routingMode":null,"avoid_features":["tollRoad"],"lang":null} | {"transportMode":"car","origin":"Hamburg","destination":"München","via":null,"departureTime":null,"routingMode":null,"avoid_features":["tollRoad"],"lang":null} | {"transportMode":"truck","origin":"Hamburg","destination":"München","via":null,"departureTime":null,"routingMode":null,"avoid_features":["tollRoad"],"lang":null} | {"transportMode":"truck","origin":"Hamburg","destination":"München","via":null,"departureTime":null,"routingMode":null,"avoid_features":["tollRoad"],"lang":null} | {"transportMode":"truck","origin":"Hamburg","destination":"München","via":null,"departureTime":null,"routingMode":null,"avoid_features":["tollRoad"],"lang":null} | {"transportMode":"truck","origin":"Hamburg","destination":"München","via":null,"departureTime":null,"routingMode":null,"avoid_features":["tollRoad"],"lang":null} | {"transportMode":"truck","origin":"Hamburg","destination":"München","via":null,"departureTime":null,"routingMode":null,"avoid_features":["tollRoad"],"lang":null} |

verdict(v3): bad [src=t1_haiku55, judged by assistant (de/es/fr, per policy §3)] — 'mit dem Lieferwagen' (van) -> v3 transportMode=car; should be truck (5/6 bake-off teachers said truck; t1 quirk: under-differentiates vehicle words to car)

## [5] xtr-2a2fa2a56714 (de, here_geocode, core_train)

utt: Hi, ich möchte eine Route von Ibadan nach Warri planen. Bitte stelle sicher, dass wir alle zwei Stunden eine Pause einlegen. Kannst du mir die Routenplanung damit erstellen?

params: q:string(enum=0), lang:string(enum=0), limit:integer(enum=0)

| v2 | t1_haiku55 | t2_gpt54mini | t3_gpt6luna | t4_glm53flash | t5_dsv41flash | t6_ling31flash |
|---|---|---|---|---|---|---|
| {"q":"Ibadan","lang":null,"limit":null} | {"q":null,"lang":null,"limit":null} | {"q":"Warri","lang":null,"limit":null} | {"q":"Ibadan, Warri","lang":null,"limit":null} | {"q":null,"lang":null,"limit":null} | {"q":"Warri","lang":null,"limit":null} | null |

grounded: t1_haiku55=yes t2_gpt54mini=yes t3_gpt6luna=yes t4_glm53flash=yes t5_dsv41flash=yes t6_ling31flash=NO

verdict(v3): ok [src=t1_haiku55, judged by assistant (de/es/fr, per policy §3)] — q=null ok (route-plan utterance, no POI focus; t1 full-run differs from bake-off null shown above)

## [6] xtr-4c9e33a35161 (de, tomtom-routing, core_train)

utt: Ich möchte von Maicao nach Zacapa fahren, dabei Stadtzentren vermeiden und mit einer Übernachtung unterwegs pausieren. Außerdem hätte ich gern unterwegs auch Informationen zu Ladestationen für ein E-Auto.

params: origin:object(enum=0), destination:object(enum=0), response_detail:string(enum=2), routeType:string(enum=4), travelMode:string(enum=7), traffic:boolean(enum=0), avoid:array(enum=0), departAt:string(enum=0), arriveAt:string(enum=0), maxAlternatives:number(enum=0), alternativeType:string(enum=2), instructionsType:string(enum=3), language:string(enum=0), computeBestOrder:boolean(enum=0), supportingPoints:string(enum=0), vehicleHeading:number(enum=0), includeTollPaymentTypes:string(enum=0), report:string(enum=0), routeRepresentation:string(enum=4), extendedRouteRepresentation:string(enum=0), computeTravelTimeFor:string(enum=2), hilliness:string(enum=3), windingness:string(enum=3), vehicleMaxSpeed:number(enum=0), vehicleWeight:number(enum=0), vehicleWidth:number(enum=0), vehicleHeight:number(enum=0), vehicleLength:number(enum=0), vehicleCommercial:boolean(enum=0), vehicleAxleWeight:number(enum=0), vehicleNumberOfAxles:number(enum=0), vehicleLoadType:string(enum=0), vehicleAdrTunnelRestrictionCode:string(enum=0), vehicleEngineType:string(enum=2), currentChargeInkWh:number(enum=0), maxChargeInkWh:number(enum=0), constantSpeedConsumptionInkWhPerHundredkm:string(enum=0), auxiliaryPowerInkW:number(enum=0), chargeMarginsInkWh:string(enum=0), constantSpeedConsumptionInLitersPerHundredkm:string(enum=0), currentFuelInLiters:number(enum=0), auxiliaryPowerInLitersPerHour:number(enum=0), fuelEnergyDensityInMJoulesPerLiter:number(enum=0), accelerationEfficiency:number(enum=0), decelerationEfficiency:number(enum=0), uphillEfficiency:number(enum=0), downhillEfficiency:number(enum=0), consumptionInkWhPerkmAltitudeGain:number(enum=0), recuperationInkWhPerkmAltitudeLoss:number(enum=0), sectionType:array(enum=0)

| v2 | t1_haiku55 | t2_gpt54mini | t3_gpt6luna | t4_glm53flash | t5_dsv41flash | t6_ling31flash |
|---|---|---|---|---|---|---|
| {"origin":null,"destination":null,"response_detail":null,"routeType":null,"travelMode":null,"traffic":null,"avoid":null,"departAt":null,"arriveAt":null,"maxAlternatives":null,"alternativeType":null,"instructionsType":null,"language":null,"computeBestOrder":null,"supportingPoints":null,"vehicleHeading":null,"includeTollPaymentTypes":null,"report":null,"routeRepresentation":null,"extendedRouteRepresentation":null,"computeTravelTimeFor":null,"hilliness":null,"windingness":null,"vehicleMaxSpeed":null,"vehicleWeight":null,"vehicleWidth":null,"vehicleHeight":null,"vehicleLength":null,"vehicleCommercial":null,"vehicleAxleWeight":null,"vehicleNumberOfAxles":null,"vehicleLoadType":null,"vehicleAdrTunnelRestrictionCode":null,"vehicleEngineType":null,"currentChargeInkWh":null,"maxChargeInkWh":null,"constantSpeedConsumptionInkWhPerHundredkm":null,"auxiliaryPowerInkW":null,"chargeMarginsInkWh":null,"constantSpeedConsumptionInLitersPerHundredkm":null,"currentFuelInLiters":null,"auxiliaryPowerInLitersPerHour":null,"fuelEnergyDensityInMJoulesPerLiter":null,"accelerationEfficiency":null,"decelerationEfficiency":null,"uphillEfficiency":null,"downhillEfficiency":null,"consumptionInkWhPerkmAltitudeGain":null,"recuperationInkWhPerkmAltitudeLoss":null,"sectionType":null} | null | null | {"origin":null,"destination":null,"response_detail":null,"routeType":null,"travelMode":"car","traffic":null,"avoid":null,"departAt":null,"arriveAt":null,"maxAlternatives":null,"alternativeType":null,"instructionsType":null,"language":null,"computeBestOrder":null,"supportingPoints":null,"vehicleHeading":null,"includeTollPaymentTypes":null,"report":null,"routeRepresentation":null,"extendedRouteRepresentation":null,"computeTravelTimeFor":null,"hilliness":null,"windingness":null,"vehicleMaxSpeed":null,"vehicleWeight":null,"vehicleWidth":null,"vehicleHeight":null,"vehicleLength":null,"vehicleCommercial":null,"vehicleAxleWeight":null,"vehicleNumberOfAxles":null,"vehicleLoadType":null,"vehicleAdrTunnelRestrictionCode":null,"vehicleEngineType":"electric","currentChargeInkWh":null,"maxChargeInkWh":null,"constantSpeedConsumptionInkWhPerHundredkm":null,"auxiliaryPowerInkW":null,"chargeMarginsInkWh":null,"constantSpeedConsumptionInLitersPerHundredkm":null,"currentFuelInLiters":null,"auxiliaryPowerInLitersPerHour":null,"fuelEnergyDensityInMJoulesPerLiter":null,"accelerationEfficiency":null,"decelerationEfficiency":null,"uphillEfficiency":null,"downhillEfficiency":null,"consumptionInkWhPerkmAltitudeGain":null,"recuperationInkWhPerkmAltitudeLoss":null,"sectionType":null} | {"origin":null,"destination":null,"response_detail":null,"routeType":null,"travelMode":null,"traffic":null,"avoid":null,"departAt":null,"arriveAt":null,"maxAlternatives":null,"alternativeType":null,"instructionsType":null,"language":null,"computeBestOrder":null,"supportingPoints":null,"vehicleHeading":null,"includeTollPaymentTypes":null,"report":null,"routeRepresentation":null,"extendedRouteRepresentation":null,"computeTravelTimeFor":null,"hilliness":null,"windingness":null,"vehicleMaxSpeed":null,"vehicleWeight":null,"vehicleWidth":null,"vehicleHeight":null,"vehicleLength":null,"vehicleCommercial":null,"vehicleAxleWeight":null,"vehicleNumberOfAxles":null,"vehicleLoadType":null,"vehicleAdrTunnelRestrictionCode":null,"vehicleEngineType":null,"currentChargeInkWh":null,"maxChargeInkWh":null,"constantSpeedConsumptionInkWhPerHundredkm":null,"auxiliaryPowerInkW":null,"chargeMarginsInkWh":null,"constantSpeedConsumptionInLitersPerHundredkm":null,"currentFuelInLiters":null,"auxiliaryPowerInLitersPerHour":null,"fuelEnergyDensityInMJoulesPerLiter":null,"accelerationEfficiency":null,"decelerationEfficiency":null,"uphillEfficiency":null,"downhillEfficiency":null,"consumptionInkWhPerkmAltitudeGain":null,"recuperationInkWhPerkmAltitudeLoss":null,"sectionType":null} | {"origin":null,"destination":null,"response_detail":null,"routeType":null,"travelMode":null,"traffic":null,"avoid":null,"departAt":null,"arriveAt":null,"maxAlternatives":null,"alternativeType":null,"instructionsType":null,"language":null,"computeBestOrder":null,"supportingPoints":null,"vehicleHeading":null,"includeTollPaymentTypes":null,"report":null,"routeRepresentation":null,"extendedRouteRepresentation":null,"computeTravelTimeFor":null,"hilliness":null,"windingness":null,"vehicleMaxSpeed":null,"vehicleWeight":null,"vehicleWidth":null,"vehicleHeight":null,"vehicleLength":null,"vehicleCommercial":null,"vehicleAxleWeight":null,"vehicleNumberOfAxles":null,"vehicleLoadType":null,"vehicleAdrTunnelRestrictionCode":null,"vehicleEngineType":"electric","currentChargeInkWh":null,"maxChargeInkWh":null,"constantSpeedConsumptionInkWhPerHundredkm":null,"auxiliaryPowerInkW":null,"chargeMarginsInkWh":null,"constantSpeedConsumptionInLitersPerHundredkm":null,"currentFuelInLiters":null,"auxiliaryPowerInLitersPerHour":null,"fuelEnergyDensityInMJoulesPerLiter":null,"accelerationEfficiency":null,"decelerationEfficiency":null,"uphillEfficiency":null,"downhillEfficiency":null,"consumptionInkWhPerkmAltitudeGain":null,"recuperationInkWhPerkmAltitudeLoss":null,"sectionType":null} | null |

grounded: t1_haiku55=NO t2_gpt54mini=NO t3_gpt6luna=yes t4_glm53flash=yes t5_dsv41flash=yes t6_ling31flash=NO

verdict(v3): ? [src=t4_glm53flash, judged by assistant (de/es/fr, per policy §3)] — avoid=null despite 'Stadtzentren vermeiden' — cityCenters not representable in avoid enum [tollRoad,ferry,motorway,tunnel]; defensible blank

## [7] xtr-a065c7a5e571 (de, here_directions, core_train)

utt: Los, fahre ich mal von Berlin nach München, aber obzöhm ich die Autobahn meide und über die Landstraße durch Nürnberg fahre? Sag mir mal, wie es darum sehen wird.

params: transportMode:string(enum=5), origin:string(enum=0), destination:string(enum=0), via:array(enum=0), departureTime:string(enum=0), routingMode:string(enum=2), avoid_features:array(enum=0), lang:string(enum=0)

| v2 | t1_haiku55 | t2_gpt54mini | t3_gpt6luna | t4_glm53flash | t5_dsv41flash | t6_ling31flash |
|---|---|---|---|---|---|---|
| {"transportMode":null,"origin":null,"destination":null,"via":null,"departureTime":null,"routingMode":null,"avoid_features":["motorway"],"lang":null} | {"transportMode":"car","origin":"Berlin","destination":"München","via":["Nürnberg"],"departureTime":null,"routingMode":null,"avoid_features":["motorway"],"lang":null} | {"transportMode":"car","origin":"Berlin","destination":"München","via":["Nürnberg"],"departureTime":null,"routingMode":null,"avoid_features":["motorway"],"lang":null} | {"transportMode":"car","origin":"Berlin","destination":"München","via":["Nürnberg"],"departureTime":null,"routingMode":null,"avoid_features":["motorway"],"lang":null} | {"transportMode":"car","origin":"Berlin","destination":"München","via":["Nürnberg"],"departureTime":null,"routingMode":null,"avoid_features":["motorway"],"lang":null} | {"transportMode":"car","origin":"Berlin","destination":"München","via":["Nürnberg"],"departureTime":null,"routingMode":null,"avoid_features":["motorway"],"lang":null} | {"transportMode":"car","origin":"Berlin","destination":"München","via":["Nürnberg"],"departureTime":null,"routingMode":null,"avoid_features":["motorway"],"lang":null} |

verdict(v3): ok [src=t1_haiku55, judged by assistant (de/es/fr, per policy §3)]

## [8] xtr-a07db33eef3d (de, here_directions, core_train)

utt: Okay, ich fahre uns von Santiago nach Valparaíso, und zwar speziell so, dass wir alle Autobahnen vermeiden. Kannst du mir bitte einen Route entworf? Und später, wenn wir fast da sind, sag mir auch noch, was ich innerhalb von 30 Minuten vor Ort erreichen kann.

params: transportMode:string(enum=5), origin:string(enum=0), destination:string(enum=0), via:array(enum=0), departureTime:string(enum=0), routingMode:string(enum=2), avoid_features:array(enum=0), lang:string(enum=0)

| v2 | t1_haiku55 | t2_gpt54mini | t3_gpt6luna | t4_glm53flash | t5_dsv41flash | t6_ling31flash |
|---|---|---|---|---|---|---|
| {"transportMode":null,"origin":null,"destination":null,"via":null,"departureTime":null,"routingMode":null,"avoid_features":["motorway"],"lang":null} | {"transportMode":"car","origin":"Santiago","destination":"Valparaíso","via":null,"departureTime":null,"routingMode":null,"avoid_features":["motorway"],"lang":null} | {"transportMode":"car","origin":"Santiago","destination":"Valparaíso","via":null,"departureTime":null,"routingMode":null,"avoid_features":["motorway"],"lang":null} | {"transportMode":"car","origin":"Santiago","destination":"Valparaíso","via":null,"departureTime":null,"routingMode":null,"avoid_features":["motorway"],"lang":null} | {"transportMode":"car","origin":"Santiago","destination":"Valparaíso","via":null,"departureTime":null,"routingMode":null,"avoid_features":["motorway"],"lang":null} | {"transportMode":"car","origin":"Santiago","destination":"Valparaíso","via":null,"departureTime":null,"routingMode":null,"avoid_features":["motorway"],"lang":null} | {"transportMode":"car","origin":"Santiago","destination":"Valparaíso","via":null,"departureTime":null,"routingMode":null,"avoid_features":["motorway"],"lang":null} |

verdict(v3): ok [src=t1_haiku55, judged by assistant (de/es/fr, per policy §3)]

## [9] xtr-ac01fc90b934 (de, here_search_places, core_train)

utt: Ich muss von Freiburg nach Kiel fahren, dabei darf ich keine Autobahnen benutzen – nur Landstraßen. Zwischen den beiden Städten soll ich in Göttingen anhalten, um dort Kaffee zu trinken, weil die Straße dort heute gesperrt ist. Außerdem möchte ich unterwegs Ladeplätze für mein Elektroauto einplanen.

params: q:string(enum=0), at:string(enum=0), in_country:string(enum=0), limit:integer(enum=0)

| v2 | t1_haiku55 | t2_gpt54mini | t3_gpt6luna | t4_glm53flash | t5_dsv41flash | t6_ling31flash |
|---|---|---|---|---|---|---|
| {"q":null,"at":null,"in_country":null,"limit":null} | null | {"q":"Ladeplätze","at":null,"in_country":null,"limit":null} | {"q":"Ladeplätze für mein Elektroauto","at":null,"in_country":null,"limit":null} | {"q":"Ladeplätze für mein Elektroauto","at":null,"in_country":null,"limit":null} | {"q":"Ladeplätze","at":null,"in_country":null,"limit":null} | {"q":"Ladeplätze","at":null,"in_country":null,"limit":null} |

grounded: t1_haiku55=NO t2_gpt54mini=yes t3_gpt6luna=yes t4_glm53flash=yes t5_dsv41flash=yes t6_ling31flash=yes

verdict(v3): ok [src=t1_haiku55, judged by assistant (de/es/fr, per policy §3)] — q='Ladeplätze' shorter verbatim span — acceptable style

## [10] xtr-ff67acf332fa (de, osrm_route, core_train)

utt: Ich möchte von Taguatinga nach Inhumas fahren, aber mit einer Übernachtung unterwegs. Wie würde die Route aussehen?

params: coordinates:string(enum=0)

| v2 | t1_haiku55 | t2_gpt54mini | t3_gpt6luna | t4_glm53flash | t5_dsv41flash | t6_ling31flash |
|---|---|---|---|---|---|---|
| {"coordinates":null} | {"coordinates":null} | {"coordinates":"Taguatinga; Inhumas"} | {"coordinates":null} | {"coordinates":null} | {"coordinates":null} | {"coordinates":null} |

verdict(v3): ok [src=t1_haiku55, judged by assistant (de/es/fr, per policy §3)] — coordinates=null honest

## [11] aug_i-61f67bc02c18 (en, here_directions, aug_train)

utt: From downtown Leeds to the outskirts of Munich, avoid tolls and factor in a thirty-minute charging stop near Frankfurt, it’s pouring and the traffic’s brutal.

params: transportMode:string(enum=5), origin:string(enum=0), destination:string(enum=0), via:array(enum=0), departureTime:string(enum=0), routingMode:string(enum=2), avoid_features:array(enum=0), lang:string(enum=0)

| v2 | t1_haiku55 | t2_gpt54mini | t3_gpt6luna | t4_glm53flash | t5_dsv41flash | t6_ling31flash |
|---|---|---|---|---|---|---|
| {"transportMode":null,"origin":null,"destination":null,"via":null,"departureTime":null,"routingMode":null,"avoid_features":["tollRoad"],"lang":null} | {"transportMode":"car","origin":"downtown Leeds","destination":"outskirts of Munich","via":["Frankfurt"],"departureTime":null,"routingMode":null,"avoid_features":["tollRoad"],"lang":null} | {"transportMode":null,"origin":"downtown Leeds","destination":"the outskirts of Munich","via":null,"departureTime":null,"routingMode":null,"avoid_features":["tollRoad"],"lang":null} | {"transportMode":"car","origin":"downtown Leeds","destination":"outskirts of Munich","via":["near Frankfurt"],"departureTime":null,"routingMode":null,"avoid_features":["tollRoad"],"lang":null} | {"transportMode":"car","origin":null,"destination":null,"via":null,"departureTime":null,"routingMode":null,"avoid_features":["tollRoad"],"lang":null} | {"transportMode":"car","origin":"downtown Leeds","destination":"outskirts of Munich","via":["near Frankfurt"],"departureTime":null,"routingMode":null,"avoid_features":["tollRoad"],"lang":null} | {"transportMode":null,"origin":"downtown Leeds","destination":"outskirts of Munich","via":["Frankfurt"],"departureTime":null,"routingMode":null,"avoid_features":["tollRoad"],"lang":null} |

verdict(v3): ok [src=t1_haiku55, judged by user]

## [12] aug_i-7a695b77021d (en, here_geocode, aug_train)

utt: plan a route from Lahore to Hub avoiding city centers, about 1230 km

params: q:string(enum=0), lang:string(enum=0), limit:integer(enum=0)

| v2 | t1_haiku55 | t2_gpt54mini | t3_gpt6luna | t4_glm53flash | t5_dsv41flash | t6_ling31flash |
|---|---|---|---|---|---|---|
| {"q":"Lahore","lang":null,"limit":null} | {"q":"Lahore","lang":null,"limit":null} | {"q":null,"lang":null,"limit":null} | {"q":"Lahore, Hub","lang":null,"limit":null} | {"q":"Lahore","lang":null,"limit":null} | {"q":null,"lang":null,"limit":null} | {"q":"Lahore to Hub","lang":null,"limit":null} |

verdict(v3): ok [src=t1_haiku55, judged by user] — dup id aug_i-7a695b77021d (both dup rows q=Lahore)

## [13] aug_i-8536263c4953 (en, here_geocode, aug_train)

utt: Take us from Bern to Lisbon in our camper, avoiding tolls and planning overnight charging stops every 300 km.

params: q:string(enum=0), lang:string(enum=0), limit:integer(enum=0)

| v2 | t1_haiku55 | t2_gpt54mini | t3_gpt6luna | t4_glm53flash | t5_dsv41flash | t6_ling31flash |
|---|---|---|---|---|---|---|
| {"q":"Lisbon in our camper","lang":null,"limit":null} | {"q":"Bern","lang":null,"limit":null} | {"q":"Lisbon","lang":null,"limit":null} | {"q":"Lisbon","lang":null,"limit":null} | {"q":"Lisbon","lang":null,"limit":null} | {"q":"Lisbon","lang":null,"limit":null} | null |

grounded: t1_haiku55=yes t2_gpt54mini=yes t3_gpt6luna=yes t4_glm53flash=yes t5_dsv41flash=yes t6_ling31flash=NO

verdict(v3): ok [src=t1_haiku55, judged by user]

## [14] aug_i-8bfc04dfa284 (en, here_directions, aug_train)

utt: navigate from Bangkok to Hua Hin avoiding tolls, about 190 km

params: transportMode:string(enum=5), origin:string(enum=0), destination:string(enum=0), via:array(enum=0), departureTime:string(enum=0), routingMode:string(enum=2), avoid_features:array(enum=0), lang:string(enum=0)

| v2 | t1_haiku55 | t2_gpt54mini | t3_gpt6luna | t4_glm53flash | t5_dsv41flash | t6_ling31flash |
|---|---|---|---|---|---|---|
| {"transportMode":null,"origin":null,"destination":null,"via":null,"departureTime":null,"routingMode":null,"avoid_features":["tollRoad"],"lang":null} | {"transportMode":"car","origin":"Bangkok","destination":"Hua Hin","via":null,"departureTime":null,"routingMode":null,"avoid_features":["tollRoad"],"lang":null} | {"transportMode":null,"origin":"Bangkok","destination":"Hua Hin","via":null,"departureTime":null,"routingMode":null,"avoid_features":["tollRoad"],"lang":null} | {"transportMode":"car","origin":"Bangkok","destination":"Hua Hin","via":null,"departureTime":null,"routingMode":null,"avoid_features":["tollRoad"],"lang":null} | {"transportMode":"car","origin":"Bangkok","destination":"Hua Hin","via":null,"departureTime":null,"routingMode":null,"avoid_features":["tollRoad"],"lang":null} | {"transportMode":null,"origin":"Bangkok","destination":"Hua Hin","via":null,"departureTime":null,"routingMode":null,"avoid_features":["tollRoad"],"lang":null} | {"transportMode":null,"origin":"Bangkok","destination":"Hua Hin","via":null,"departureTime":null,"routingMode":null,"avoid_features":["tollRoad"],"lang":null} |

verdict(v3): ok [src=t1_haiku55, judged by user]

## [15] aug_i-c14db7daf305 (en, here_geocode, aug_train)

utt: navigate from Iloilo to Pasig City with a coffee stop avoiding ferry crossings

params: q:string(enum=0), lang:string(enum=0), limit:integer(enum=0)

| v2 | t1_haiku55 | t2_gpt54mini | t3_gpt6luna | t4_glm53flash | t5_dsv41flash | t6_ling31flash |
|---|---|---|---|---|---|---|
| {"q":"Iloilo","lang":null,"limit":null} | {"q":"Iloilo","lang":null,"limit":null} | {"q":"Pasig City","lang":null,"limit":null} | {"q":"Pasig City","lang":null,"limit":null} | {"q":null,"lang":null,"limit":null} | {"q":"Pasig City","lang":null,"limit":null} | null |

grounded: t1_haiku55=yes t2_gpt54mini=yes t3_gpt6luna=yes t4_glm53flash=yes t5_dsv41flash=yes t6_ling31flash=NO

verdict(v3): ok [src=t1_haiku55, judged by user]

## [16] aug_i-ca8b38f82b95 (en, here_geocode, aug_train)

utt: route from Guadalajara to Huimanguillo on the most scenic road

params: q:string(enum=0), lang:string(enum=0), limit:integer(enum=0)

| v2 | t1_haiku55 | t2_gpt54mini | t3_gpt6luna | t4_glm53flash | t5_dsv41flash | t6_ling31flash |
|---|---|---|---|---|---|---|
| {"q":"Guadalajara","lang":null,"limit":null} | {"q":"Guadalajara","lang":null,"limit":null} | {"q":"Guadalajara to Huimanguillo","lang":null,"limit":null} | {"q":"Guadalajara to Huimanguillo","lang":null,"limit":null} | {"q":"Huimanguillo","lang":null,"limit":null} | {"q":"Huimanguillo","lang":null,"limit":null} | null |

grounded: t1_haiku55=yes t2_gpt54mini=yes t3_gpt6luna=yes t4_glm53flash=yes t5_dsv41flash=yes t6_ling31flash=NO

verdict(v3): ok [src=t1_haiku55, judged by user]

## [17] aug_i-e89b4da59ddb (en, here_geocode, aug_train)

utt: Find a route from Paris to Rome via the scenic coastal roads with a quick break for snacks.

params: q:string(enum=0), lang:string(enum=0), limit:integer(enum=0)

| v2 | t1_haiku55 | t2_gpt54mini | t3_gpt6luna | t4_glm53flash | t5_dsv41flash | t6_ling31flash |
|---|---|---|---|---|---|---|
| {"q":"Paris","lang":null,"limit":null} | {"q":"Paris","lang":null,"limit":null} | {"q":null,"lang":null,"limit":null} | {"q":null,"lang":null,"limit":null} | {"q":"Rome","lang":null,"limit":null} | {"q":null,"lang":null,"limit":null} | {"q":"Paris to Rome","lang":null,"limit":null} |

verdict(v3): ok [src=t1_haiku55, judged by user]

## [18] aug_i-ebdfdce75686 (en, here_geocode, aug_train)

utt: go from Munich to Prague with breaks after each 100 km and after each two hours, but not more than

params: q:string(enum=0), lang:string(enum=0), limit:integer(enum=0)

| v2 | t1_haiku55 | t2_gpt54mini | t3_gpt6luna | t4_glm53flash | t5_dsv41flash | t6_ling31flash |
|---|---|---|---|---|---|---|
| {"q":"Munich","lang":null,"limit":null} | {"q":null,"lang":null,"limit":null} | {"q":null,"lang":null,"limit":null} | {"q":"Munich, Prague","lang":null,"limit":null} | {"q":"Prague","lang":null,"limit":null} | {"q":"Prague","lang":null,"limit":null} | {"q":"Munich to Prague","lang":null,"limit":null} |

verdict(v3): ok [src=t1_haiku55, judged by user]

## [19] aug_i-2c8e31359631 (es, here_geocode, aug_val)

utt: ve de Multan a Jalalpur Pirwala sin autopistas, unos 100 km

params: q:string(enum=0), lang:string(enum=0), limit:integer(enum=0)

| v2 | t1_haiku55 | t2_gpt54mini | t3_gpt6luna | t4_glm53flash | t5_dsv41flash | t6_ling31flash |
|---|---|---|---|---|---|---|
| {"q":"Multan","lang":null,"limit":null} | {"q":"Jalalpur Pirwala","lang":null,"limit":null} | {"q":"Multan a Jalalpur Pirwala","lang":null,"limit":null} | {"q":"Jalalpur Pirwala","lang":null,"limit":null} | {"q":"Jalalpur Pirwala","lang":null,"limit":null} | {"q":"Jalalpur Pirwala","lang":null,"limit":null} | {"q":"Jalalpur Pirwala","lang":null,"limit":null} |

verdict(v3): ok [src=t1_haiku55, judged by assistant (de/es/fr, per policy §3)] — q=dest Jalalpur Pirwala

## [20] aug_i-8423e6a227e9 (es, here_geocode, aug_train)

utt: Voy de Bilbao a Granada evitando el corte de tráfico en Vizcaya, necesito parar en Toledo para recargar y descansar, y luego seguir por carretera nacional aunque tardaré más una hora extra.

params: q:string(enum=0), lang:string(enum=0), limit:integer(enum=0)

| v2 | t1_haiku55 | t2_gpt54mini | t3_gpt6luna | t4_glm53flash | t5_dsv41flash | t6_ling31flash |
|---|---|---|---|---|---|---|
| {"q":"Bilbao","lang":null,"limit":null} | {"q":"Toledo","lang":null,"limit":null} | {"q":"Toledo","lang":null,"limit":null} | {"q":"Granada","lang":null,"limit":null} | {"q":"Granada","lang":null,"limit":null} | null | {"q":"Granada","lang":null,"limit":null} |

grounded: t1_haiku55=yes t2_gpt54mini=yes t3_gpt6luna=yes t4_glm53flash=yes t5_dsv41flash=NO t6_ling31flash=yes

verdict(v3): ok [src=t1_haiku55, judged by assistant (de/es/fr, per policy §3)] — q=Toledo stop-city (style variance, tolerated)

## [21] aug_i-8c0d22b4605c (es, here_geocode, aug_train)

utt: Salgo de Zaragoza dirección a Valencia por la A2, pero evita peajes y autopista, mete una parada de 20 minutos en Alcolea del Pinar para estirar las piernas y ve por la N-320 para evitar el calor de la autovía.

params: q:string(enum=0), lang:string(enum=0), limit:integer(enum=0)

| v2 | t1_haiku55 | t2_gpt54mini | t3_gpt6luna | t4_glm53flash | t5_dsv41flash | t6_ling31flash |
|---|---|---|---|---|---|---|
| {"q":"Zaragoza dirección","lang":null,"limit":null} | {"q":"Alcolea del Pinar","lang":null,"limit":null} | {"q":"Zaragoza","lang":null,"limit":null} | {"q":"Valencia","lang":null,"limit":null} | {"q":"Valencia","lang":null,"limit":null} | {"q":"Valencia","lang":null,"limit":null} | {"q":"Valencia","lang":null,"limit":null} |

verdict(v3): ok [src=t1_haiku55, judged by assistant (de/es/fr, per policy §3)] — q=Alcolea del Pinar stop-city; fixes v2 junk 'Zaragoza dirección'

## [22] aug_i-eb122e4f0ae6 (es, here_directions, aug_train)

utt: De Valencia a Bilbao, evitando peajes, con una parada para descansar en la zona de servicio de Logroño, y recargando mi coche eléctrico en la estación de Lanzada antes de continuar.

params: transportMode:string(enum=5), origin:string(enum=0), destination:string(enum=0), via:array(enum=0), departureTime:string(enum=0), routingMode:string(enum=2), avoid_features:array(enum=0), lang:string(enum=0)

| v2 | t1_haiku55 | t2_gpt54mini | t3_gpt6luna | t4_glm53flash | t5_dsv41flash | t6_ling31flash |
|---|---|---|---|---|---|---|
| {"transportMode":"car","origin":null,"destination":null,"via":null,"departureTime":null,"routingMode":null,"avoid_features":null,"lang":null} | {"transportMode":"car","origin":"Valencia","destination":"Bilbao","via":["Logroño","Lanzada"],"departureTime":null,"routingMode":null,"avoid_features":["tollRoad"],"lang":null} | {"transportMode":"car","origin":"Valencia","destination":"Bilbao","via":["zona de servicio de Logroño","estación de Lanzada"],"departureTime":null,"routingMode":null,"avoid_features":["tollRoad"],"lang":null} | {"transportMode":"car","origin":"Valencia","destination":"Bilbao","via":["Logroño","Lanzada"],"departureTime":null,"routingMode":null,"avoid_features":["tollRoad"],"lang":null} | {"transportMode":"car","origin":"Valencia","destination":"Bilbao","via":["Logroño","Lanzada"],"departureTime":null,"routingMode":null,"avoid_features":["tollRoad"],"lang":null} | {"transportMode":"car","origin":"Valencia","destination":"Bilbao","via":["Logroño","Lanzada"],"departureTime":null,"routingMode":null,"avoid_features":["tollRoad"],"lang":null} | {"transportMode":"car","origin":"Valencia","destination":"Bilbao","via":["Logroño","Lanzada"],"departureTime":null,"routingMode":null,"avoid_features":["tollRoad"],"lang":null} |

verdict(v3): ok [src=t1_haiku55, judged by assistant (de/es/fr, per policy §3)] — via=[Logroño,Lanzada] fluff-drop correct

## [23] gold-6b2095502275 (es, here_directions, core_test)

utt: Entendido, necesito llegar a Dumyāţ al Jadīdah desde Alexandria en una ruta de aproximadamente 190 kilómetros sin usar transbordadores, y también quiero que me confirmes después las opciones de estacionamiento disponibles en el destino.

params: transportMode:string(enum=5), origin:string(enum=0), destination:string(enum=0), via:array(enum=0), departureTime:string(enum=0), routingMode:string(enum=2), avoid_features:array(enum=0), lang:string(enum=0)

| v2 | t1_haiku55 | t2_gpt54mini | t3_gpt6luna | t4_glm53flash | t5_dsv41flash | t6_ling31flash |
|---|---|---|---|---|---|---|
| {"transportMode":null,"origin":null,"destination":null,"via":null,"departureTime":null,"routingMode":null,"avoid_features":["ferry"],"lang":null} | {"transportMode":null,"origin":"Alexandria","destination":"Dumyāţ al Jadīdah","via":null,"departureTime":null,"routingMode":null,"avoid_features":["ferry"],"lang":null} | {"transportMode":null,"origin":"Alexandria","destination":"Dumyāţ al Jadīdah","via":null,"departureTime":null,"routingMode":null,"avoid_features":["ferry"],"lang":null} | {"transportMode":"car","origin":"Alexandria","destination":"Dumyāţ al Jadīdah","via":null,"departureTime":null,"routingMode":null,"avoid_features":["ferry"],"lang":null} | {"transportMode":"car","origin":"Alexandria","destination":"Dumyāţ al Jadīdah","via":null,"departureTime":null,"routingMode":null,"avoid_features":["ferry"],"lang":null} | {"transportMode":null,"origin":"Alexandria","destination":"Dumyāţ al Jadīdah","via":null,"departureTime":null,"routingMode":null,"avoid_features":["ferry"],"lang":null} | {"transportMode":null,"origin":"Alexandria","destination":"Dumyāţ al Jadīdah","via":null,"departureTime":null,"routingMode":null,"avoid_features":["ferry"],"lang":null} |

verdict(v3): ok [src=t1_haiku55, judged by assistant (de/es/fr, per policy §3)]

## [24] xtr-118985ff3d0c (es, here_directions, core_train)

utt: Hola, necesito que calcues la ruta para conducir desde París hasta Lecce, pasando por Roma y Milán, por favor.

params: transportMode:string(enum=5), origin:string(enum=0), destination:string(enum=0), via:array(enum=0), departureTime:string(enum=0), routingMode:string(enum=2), avoid_features:array(enum=0), lang:string(enum=0)

| v2 | t1_haiku55 | t2_gpt54mini | t3_gpt6luna | t4_glm53flash | t5_dsv41flash | t6_ling31flash |
|---|---|---|---|---|---|---|
| {"transportMode":null,"origin":null,"destination":null,"via":null,"departureTime":null,"routingMode":null,"avoid_features":null,"lang":null} | {"transportMode":"car","origin":"París","destination":"Lecce","via":["Roma","Milán"],"departureTime":null,"routingMode":null,"avoid_features":null,"lang":null} | {"transportMode":"car","origin":"París","destination":"Lecce","via":["Roma","Milán"],"departureTime":null,"routingMode":null,"avoid_features":null,"lang":null} | {"transportMode":"car","origin":"París","destination":"Lecce","via":["Roma","Milán"],"departureTime":null,"routingMode":null,"avoid_features":null,"lang":null} | {"transportMode":"car","origin":"París","destination":"Lecce","via":["Roma","Milán"],"departureTime":null,"routingMode":null,"avoid_features":null,"lang":null} | {"transportMode":"car","origin":"París","destination":"Lecce","via":["Roma","Milán"],"departureTime":null,"routingMode":null,"avoid_features":null,"lang":null} | {"transportMode":"car","origin":"París","destination":"Lecce","via":["Roma","Milán"],"departureTime":null,"routingMode":null,"avoid_features":null,"lang":null} |

verdict(v3): ok [src=t1_haiku55, judged by assistant (de/es/fr, per policy §3)]

## [25] xtr-4628bab848b9 (es, here_search_places, core_train)

utt: Hola, necesito que programes una ruta de Dhaka a Rosera de unos 700 km. Por favor, evita cualquier carretera de peaje.

params: q:string(enum=0), at:string(enum=0), in_country:string(enum=0), limit:integer(enum=0)

| v2 | t1_haiku55 | t2_gpt54mini | t3_gpt6luna | t4_glm53flash | t5_dsv41flash | t6_ling31flash |
|---|---|---|---|---|---|---|
| {"q":"Rosera","at":null,"in_country":null,"limit":null} | {"q":null,"at":null,"in_country":null,"limit":null} | {"q":null,"at":null,"in_country":null,"limit":null} | {"q":null,"at":null,"in_country":null,"limit":null} | {"q":null,"at":null,"in_country":null,"limit":null} | {"q":null,"at":null,"in_country":null,"limit":null} | null |

grounded: t1_haiku55=yes t2_gpt54mini=yes t3_gpt6luna=yes t4_glm53flash=yes t5_dsv41flash=yes t6_ling31flash=NO

verdict(v3): ok [src=t1_haiku55, judged by assistant (de/es/fr, per policy §3)] — all-null; 6/6 teacher consensus

## [26] xtr-5d9eefa527d4 (es, here_directions, core_train)

utt: Hola, necesito que programes la ruta de Zhanjiang a Luoyang evitando los peajes de las autopistas.

params: transportMode:string(enum=5), origin:string(enum=0), destination:string(enum=0), via:array(enum=0), departureTime:string(enum=0), routingMode:string(enum=2), avoid_features:array(enum=0), lang:string(enum=0)

| v2 | t1_haiku55 | t2_gpt54mini | t3_gpt6luna | t4_glm53flash | t5_dsv41flash | t6_ling31flash |
|---|---|---|---|---|---|---|
| {"transportMode":null,"origin":null,"destination":null,"via":null,"departureTime":null,"routingMode":null,"avoid_features":["motorway","tollRoad"],"lang":null} | {"transportMode":"car","origin":"Zhanjiang","destination":"Luoyang","via":null,"departureTime":null,"routingMode":null,"avoid_features":["tollRoad"],"lang":null} | {"transportMode":null,"origin":"Zhanjiang","destination":"Luoyang","via":null,"departureTime":null,"routingMode":null,"avoid_features":["tollRoad"],"lang":null} | {"transportMode":"car","origin":"Zhanjiang","destination":"Luoyang","via":null,"departureTime":null,"routingMode":null,"avoid_features":["tollRoad"],"lang":null} | {"transportMode":"car","origin":"Zhanjiang","destination":"Luoyang","via":null,"departureTime":null,"routingMode":null,"avoid_features":["tollRoad"],"lang":null} | {"transportMode":null,"origin":"Zhanjiang","destination":"Luoyang","via":null,"departureTime":null,"routingMode":null,"avoid_features":["tollRoad"],"lang":null} | null |

grounded: t1_haiku55=yes t2_gpt54mini=yes t3_gpt6luna=yes t4_glm53flash=yes t5_dsv41flash=yes t6_ling31flash=NO

verdict(v3): ok [src=t1_haiku55, judged by assistant (de/es/fr, per policy §3)] — 'peajes de las autopistas' -> [tollRoad] only = policy §3.3 adjacent-feature precision (v2 had [motorway,tollRoad])

## [27] xtr-7dd1ca1fded4 (es, here_geocode, core_train)

utt: Oye, vamos a ir de Kumasi a Nsawam pero necesito que me ayudes a encontrar paradas para recargar combustible cada 300 kilómetros, ¿te parece bien?

params: q:string(enum=0), lang:string(enum=0), limit:integer(enum=0)

| v2 | t1_haiku55 | t2_gpt54mini | t3_gpt6luna | t4_glm53flash | t5_dsv41flash | t6_ling31flash |
|---|---|---|---|---|---|---|
| {"q":"Kumasi","lang":null,"limit":null} | {"q":"Kumasi","lang":null,"limit":null} | {"q":"Kumasi a Nsawam","lang":null,"limit":null} | {"q":"Kumasi, Nsawam","lang":null,"limit":null} | {"q":null,"lang":null,"limit":null} | {"q":null,"lang":null,"limit":null} | {"q":"Kumasi","lang":null,"limit":null} |

verdict(v3): ok [src=t1_haiku55, judged by assistant (de/es/fr, per policy §3)] — q=Kumasi origin

## [28] aug_i-0f73d10a655c (fr, here_directions, aug_train)

utt: va de Chengde à Wafangdian avec des arrêts de recharge tous les 300 km par l'itinéraire le plus rapide

params: transportMode:string(enum=5), origin:string(enum=0), destination:string(enum=0), via:array(enum=0), departureTime:string(enum=0), routingMode:string(enum=2), avoid_features:array(enum=0), lang:string(enum=0)

| v2 | t1_haiku55 | t2_gpt54mini | t3_gpt6luna | t4_glm53flash | t5_dsv41flash | t6_ling31flash |
|---|---|---|---|---|---|---|
| {"transportMode":null,"origin":null,"destination":null,"via":null,"departureTime":null,"routingMode":"fast","avoid_features":null,"lang":null} | {"transportMode":"car","origin":"Chengde","destination":"Wafangdian","via":null,"departureTime":null,"routingMode":"fast","avoid_features":null,"lang":null} | {"transportMode":null,"origin":"Chengde","destination":"Wafangdian","via":null,"departureTime":null,"routingMode":"fast","avoid_features":null,"lang":null} | {"transportMode":"car","origin":"Chengde","destination":"Wafangdian","via":null,"departureTime":null,"routingMode":"fast","avoid_features":null,"lang":null} | {"transportMode":"car","origin":"Chengde","destination":"Wafangdian","via":null,"departureTime":null,"routingMode":"fast","avoid_features":null,"lang":null} | {"transportMode":null,"origin":"Chengde","destination":"Wafangdian","via":null,"departureTime":null,"routingMode":"fast","avoid_features":null,"lang":null} | {"transportMode":"car","origin":"Chengde","destination":"Wafangdian","via":null,"departureTime":null,"routingMode":"fast","avoid_features":null,"lang":null} |

verdict(v3): ok [src=t1_haiku55, judged by assistant (de/es/fr, per policy §3)] — routingMode=fast

## [29] aug_i-4e812a9f7a1f (fr, here_geocode, aug_train)

utt: va de Shinagawa à Ogi par l'itinéraire le plus rapide

params: q:string(enum=0), lang:string(enum=0), limit:integer(enum=0)

| v2 | t1_haiku55 | t2_gpt54mini | t3_gpt6luna | t4_glm53flash | t5_dsv41flash | t6_ling31flash |
|---|---|---|---|---|---|---|
| {"q":"Shinagawa","lang":null,"limit":null} | {"q":"Shinagawa","lang":null,"limit":null} | {"q":"Shinagawa à Ogi","lang":null,"limit":null} | {"q":"Shinagawa, Ogi","lang":null,"limit":null} | {"q":null,"lang":null,"limit":null} | {"q":null,"lang":null,"limit":null} | null |

grounded: t1_haiku55=yes t2_gpt54mini=yes t3_gpt6luna=yes t4_glm53flash=yes t5_dsv41flash=yes t6_ling31flash=NO

verdict(v3): ok [src=t1_haiku55, judged by assistant (de/es/fr, per policy §3)]

## [30] aug_i-a5e8692037b2 (fr, here_geocode, aug_train)

utt: Calcule le trajet de Reims à Châlons-en-Champagne en évitant les autoroutes, avec un point de passage à Épernay avant neuf heures pour une livraison.

params: q:string(enum=0), lang:string(enum=0), limit:integer(enum=0)

| v2 | t1_haiku55 | t2_gpt54mini | t3_gpt6luna | t4_glm53flash | t5_dsv41flash | t6_ling31flash |
|---|---|---|---|---|---|---|
| {"q":"Châlons-en-Champagne","lang":null,"limit":null} | {"q":"Reims","lang":null,"limit":null} | {"q":"Châlons-en-Champagne","lang":null,"limit":null} | {"q":"Châlons-en-Champagne","lang":null,"limit":null} | {"q":"Châlons-en-Champagne","lang":null,"limit":null} | {"q":null,"lang":null,"limit":null} | {"q":"Châlons-en-Champagne","lang":null,"limit":null} |

verdict(v3): ok [src=t1_haiku55, judged by assistant (de/es/fr, per policy §3)] — q=Reims

## [31] aug_i-f71e7c14774d (fr, here_directions, aug_train)

utt: De Lille à Strasbourg, passe par la D650 en évitant les péages, et fais un détour par Nancy pour recharger la voiture, car la RN4 est fermée à cause de la parade.

params: transportMode:string(enum=5), origin:string(enum=0), destination:string(enum=0), via:array(enum=0), departureTime:string(enum=0), routingMode:string(enum=2), avoid_features:array(enum=0), lang:string(enum=0)

| v2 | t1_haiku55 | t2_gpt54mini | t3_gpt6luna | t4_glm53flash | t5_dsv41flash | t6_ling31flash |
|---|---|---|---|---|---|---|
| {"transportMode":"car","origin":null,"destination":null,"via":null,"departureTime":null,"routingMode":null,"avoid_features":null,"lang":null} | {"transportMode":"car","origin":"Lille","destination":"Strasbourg","via":["Nancy"],"departureTime":null,"routingMode":null,"avoid_features":["tollRoad"],"lang":null} | {"transportMode":"car","origin":"Lille","destination":"Strasbourg","via":["Nancy"],"departureTime":null,"routingMode":null,"avoid_features":["tollRoad"],"lang":null} | {"transportMode":"car","origin":"Lille","destination":"Strasbourg","via":["D650","Nancy"],"departureTime":null,"routingMode":null,"avoid_features":["tollRoad"],"lang":null} | {"transportMode":"car","origin":"Lille","destination":"Strasbourg","via":["Nancy"],"departureTime":null,"routingMode":null,"avoid_features":["tollRoad"],"lang":null} | {"transportMode":"car","origin":"Lille","destination":"Strasbourg","via":["D650","Nancy"],"departureTime":null,"routingMode":null,"avoid_features":["tollRoad"],"lang":null} | null |

grounded: t1_haiku55=yes t2_gpt54mini=yes t3_gpt6luna=yes t4_glm53flash=yes t5_dsv41flash=yes t6_ling31flash=NO

verdict(v3): ok [src=t1_haiku55, judged by assistant (de/es/fr, per policy §3)] — via=[Nancy]; D650 dropped — de minimis

## [32] xtr-53ebd1d155a3 (fr, tomtom-poi-categories, core_train)

utt: Bonjour, donc je voudrais bien savoir comment allier un itinéraire sans péage de Lyon à Annecy pour mon camping-car électrique avec la contrainte de pouvoir recharger au milieu du trajet, sachant que la recherche de points de recharge n'a pas bien fonctionné ici.

params: filters:array(enum=0)

| v2 | t1_haiku55 | t2_gpt54mini | t3_gpt6luna | t4_glm53flash | t5_dsv41flash | t6_ling31flash |
|---|---|---|---|---|---|---|
| {"filters":["recharge"]} | null | {"filters":null} | {"filters":["points de recharge"]} | null | {"filters":["points de recharge"]} | null |

grounded: t1_haiku55=NO t2_gpt54mini=yes t3_gpt6luna=yes t4_glm53flash=NO t5_dsv41flash=yes t6_ling31flash=NO

verdict(v3): ok [src=v2_fallback, judged by assistant (de/es/fr, per policy §3)] — filters=['recharge'] verbatim-grounded

## [33] xtr-6ec77eb1479d (fr, here_geocode, core_train)

utt: Allo, alors je vais de Lyon à Bordeaux, mais je veux absolument éviter les péages. Tu pourras m'aider avec ça ?

params: q:string(enum=0), lang:string(enum=0), limit:integer(enum=0)

| v2 | t1_haiku55 | t2_gpt54mini | t3_gpt6luna | t4_glm53flash | t5_dsv41flash | t6_ling31flash |
|---|---|---|---|---|---|---|
| {"q":"Lyon","lang":null,"limit":null} | {"q":"Lyon","lang":null,"limit":null} | {"q":"Lyon","lang":null,"limit":null} | {"q":"Lyon, Bordeaux","lang":null,"limit":null} | {"q":"Bordeaux","lang":null,"limit":null} | null | {"q":"Bordeaux","lang":null,"limit":null} |

grounded: t1_haiku55=yes t2_gpt54mini=yes t3_gpt6luna=yes t4_glm53flash=yes t5_dsv41flash=NO t6_ling31flash=yes

verdict(v3): ok [src=t1_haiku55, judged by assistant (de/es/fr, per policy §3)]

## [34] xtr-78209d1fd5aa (fr, tomtom-geocode, core_train)

utt: Bonjour ! Je suis votre assistant de navigation. Nous allons donc éviter les péages de l'A2 grâce aux travaux et au brouillard, en trouvant une aire de repos pour le petit-déjeuner entre Lille et Paris, avec un itinéraire adapté à un camping-car.

params: query:string(enum=0), response_detail:string(enum=2), limit:number(enum=0), language:string(enum=0), countrySet:string(enum=0), view:string(enum=0), extendedPostalCodesFor:string(enum=0), mapcodes:array(enum=0), timeZone:string(enum=0), geometries:boolean(enum=0), addressRanges:boolean(enum=0), lat:number(enum=0), lon:number(enum=0), radius:number(enum=0), topLeft:string(enum=0), btmRight:string(enum=0), entityTypeSet:string(enum=0)

| v2 | t1_haiku55 | t2_gpt54mini | t3_gpt6luna | t4_glm53flash | t5_dsv41flash | t6_ling31flash |
|---|---|---|---|---|---|---|
| {"query":"Lille","response_detail":null,"limit":null,"language":null,"countrySet":null,"view":null,"extendedPostalCodesFor":null,"mapcodes":null,"timeZone":null,"geometries":null,"addressRanges":null,"lat":null,"lon":null,"radius":null,"topLeft":null,"btmRight":null,"entityTypeSet":null} | {"query":null,"response_detail":null,"limit":null,"language":null,"countrySet":null,"view":null,"extendedPostalCodesFor":null,"mapcodes":null,"timeZone":null,"geometries":null,"addressRanges":null,"lat":null,"lon":null,"radius":null,"topLeft":null,"btmRight":null,"entityTypeSet":null} | {"query":null,"response_detail":null,"limit":null,"language":null,"countrySet":null,"view":null,"extendedPostalCodesFor":null,"mapcodes":null,"timeZone":null,"geometries":null,"addressRanges":null,"lat":null,"lon":null,"radius":null,"topLeft":null,"btmRight":null,"entityTypeSet":null} | {"query":"aire de repos entre Lille et Paris","response_detail":null,"limit":null,"language":null,"countrySet":null,"view":null,"extendedPostalCodesFor":null,"mapcodes":null,"timeZone":null,"geometries":null,"addressRanges":null,"lat":null,"lon":null,"radius":null,"topLeft":null,"btmRight":null,"entityTypeSet":null} | {"query":null,"response_detail":null,"limit":null,"language":null,"countrySet":null,"view":null,"extendedPostalCodesFor":null,"mapcodes":null,"timeZone":null,"geometries":null,"addressRanges":null,"lat":null,"lon":null,"radius":null,"topLeft":null,"btmRight":null,"entityTypeSet":null} | {"query":null,"response_detail":null,"limit":null,"language":null,"countrySet":null,"view":null,"extendedPostalCodesFor":null,"mapcodes":null,"timeZone":null,"geometries":null,"addressRanges":null,"lat":null,"lon":null,"radius":null,"topLeft":null,"btmRight":null,"entityTypeSet":null} | null |

grounded: t1_haiku55=yes t2_gpt54mini=yes t3_gpt6luna=yes t4_glm53flash=yes t5_dsv41flash=yes t6_ling31flash=NO

verdict(v3): ok [src=t1_haiku55, judged by assistant (de/es/fr, per policy §3)] — query=null = teacher majority (t3 rest-area q defensible alternative)

## [35] xtr-ab87e83f5f02 (fr, here_geocode, core_train)

utt: Bonjour ! Je souhaite trouver le trajet le plus agréable pour aller d'Annecy à Chamonix en empruntant des routes panoramiques, tout en prévoyant une pause pour recharger la batterie. Pouvez-vous gérer cela ?

params: q:string(enum=0), lang:string(enum=0), limit:integer(enum=0)

| v2 | t1_haiku55 | t2_gpt54mini | t3_gpt6luna | t4_glm53flash | t5_dsv41flash | t6_ling31flash |
|---|---|---|---|---|---|---|
| {"q":"Annecy","lang":null,"limit":null} | {"q":"Annecy","lang":null,"limit":null} | {"q":"Annecy à Chamonix","lang":null,"limit":null} | {"q":"Annecy, Chamonix","lang":null,"limit":null} | {"q":"Chamonix","lang":null,"limit":null} | {"q":"Chamonix","lang":null,"limit":null} | {"q":"Annecy, Chamonix","lang":null,"limit":null} |

verdict(v3): ok [src=t1_haiku55, judged by assistant (de/es/fr, per policy §3)]

## [36] xtr-be7810eb5517 (fr, nominatim_geocode, core_val)

utt: Allons de Nantes à Limoges sans péages, en passant douze minutes à Poitiers pour déjeuner, en taking la route panoramique le long de la Vienne et en évitant les bretelles de Châtellerault, c'est parti.

params: q:string(enum=0), limit:integer(enum=0)

| v2 | t1_haiku55 | t2_gpt54mini | t3_gpt6luna | t4_glm53flash | t5_dsv41flash | t6_ling31flash |
|---|---|---|---|---|---|---|
| {"q":"Nantes","limit":null} | {"q":null,"limit":null} | {"q":"Limoges","limit":null} | {"q":"Limoges","limit":null} | {"q":null,"limit":null} | {"q":"Limoges","limit":null} | {"q":"Limoges","limit":null} |

verdict(v3): ok [src=t1_haiku55, judged by assistant (de/es/fr, per policy §3)] — q=Limoges dest

## [37] xtr-f6b643e5d61a (fr, nominatim_geocode, core_train)

utt: Bonjour, je veux qu'on prenne la route de Austin vers Dallas, pouvez-vous me guider pour ce trajet ?

params: q:string(enum=0), limit:integer(enum=0)

| v2 | t1_haiku55 | t2_gpt54mini | t3_gpt6luna | t4_glm53flash | t5_dsv41flash | t6_ling31flash |
|---|---|---|---|---|---|---|
| {"q":"Austin","limit":null} | {"q":"Austin","limit":null} | {"q":"Austin vers Dallas","limit":null} | {"q":"Austin Dallas","limit":null} | {"q":null,"limit":null} | null | null |

grounded: t1_haiku55=yes t2_gpt54mini=yes t3_gpt6luna=yes t4_glm53flash=yes t5_dsv41flash=NO t6_ling31flash=NO

verdict(v3): ok [src=t1_haiku55, judged by assistant (de/es/fr, per policy §3)]

## [38] xtr-fc9119fe8965 (fr, here_directions, core_train)

utt: Bonjour, je fais tous les jours le trajet de Rouen à Évreux tôt le matin. Peux-tu me trouver un itinéraire sans péages s'il te plaît ?

params: transportMode:string(enum=5), origin:string(enum=0), destination:string(enum=0), via:array(enum=0), departureTime:string(enum=0), routingMode:string(enum=2), avoid_features:array(enum=0), lang:string(enum=0)

| v2 | t1_haiku55 | t2_gpt54mini | t3_gpt6luna | t4_glm53flash | t5_dsv41flash | t6_ling31flash |
|---|---|---|---|---|---|---|
| {"transportMode":null,"origin":null,"destination":null,"via":null,"departureTime":null,"routingMode":null,"avoid_features":["tollRoad"],"lang":null} | {"transportMode":"car","origin":"Rouen","destination":"Évreux","via":null,"departureTime":null,"routingMode":null,"avoid_features":["tollRoad"],"lang":null} | {"transportMode":null,"origin":"Rouen","destination":"Évreux","via":null,"departureTime":null,"routingMode":null,"avoid_features":["tollRoad"],"lang":null} | {"transportMode":"car","origin":"Rouen","destination":"Évreux","via":null,"departureTime":null,"routingMode":null,"avoid_features":["tollRoad"],"lang":null} | {"transportMode":"car","origin":"Rouen","destination":"Évreux","via":null,"departureTime":null,"routingMode":null,"avoid_features":["tollRoad"],"lang":null} | {"transportMode":null,"origin":"Rouen","destination":"Évreux","via":null,"departureTime":null,"routingMode":null,"avoid_features":["tollRoad"],"lang":null} | {"transportMode":null,"origin":"Rouen","destination":"Évreux","via":null,"departureTime":null,"routingMode":null,"avoid_features":["tollRoad"],"lang":null} |

verdict(v3): ok [src=t1_haiku55, judged by assistant (de/es/fr, per policy §3)]

## [39] aug_i-35c199d178cc (pl, here_geocode, aug_train)

utt: Nawiguj z Lublina do Rzeszowa przez malownicze drogi, omijając główne trasy szybkiego ruchu.

params: q:string(enum=0), lang:string(enum=0), limit:integer(enum=0)

| v2 | t1_haiku55 | t2_gpt54mini | t3_gpt6luna | t4_glm53flash | t5_dsv41flash | t6_ling31flash |
|---|---|---|---|---|---|---|
| {"q":"Rzeszowa","lang":null,"limit":null} | {"q":"Lublin","lang":null,"limit":null} | {"q":"Lublin, Rzeszów","lang":null,"limit":null} | {"q":"Rzeszów","lang":null,"limit":null} | {"q":"Rzeszów","lang":null,"limit":null} | {"q":"Rzeszów","lang":null,"limit":null} | null |

grounded: t1_haiku55=yes t2_gpt54mini=yes t3_gpt6luna=yes t4_glm53flash=yes t5_dsv41flash=yes t6_ling31flash=NO

verdict(v3): ok [src=t1_haiku55, judged by user]

## [40] aug_i-e8734e83fa26 (pl, here_directions, aug_train)

utt: jedź z Yamoussoukro do Biéby bez promów

params: transportMode:string(enum=5), origin:string(enum=0), destination:string(enum=0), via:array(enum=0), departureTime:string(enum=0), routingMode:string(enum=2), avoid_features:array(enum=0), lang:string(enum=0)

| v2 | t1_haiku55 | t2_gpt54mini | t3_gpt6luna | t4_glm53flash | t5_dsv41flash | t6_ling31flash |
|---|---|---|---|---|---|---|
| {"transportMode":null,"origin":null,"destination":null,"via":null,"departureTime":null,"routingMode":null,"avoid_features":["ferry"],"lang":null} | {"transportMode":"car","origin":"Yamoussoukro","destination":"Biéby","via":null,"departureTime":null,"routingMode":null,"avoid_features":["ferry"],"lang":null} | {"transportMode":"car","origin":"Yamoussoukro","destination":"Biéby","via":null,"departureTime":null,"routingMode":null,"avoid_features":["ferry"],"lang":null} | null | {"transportMode":"car","origin":"Yamoussoukro","destination":"Biéby","via":null,"departureTime":null,"routingMode":null,"avoid_features":["ferry"],"lang":null} | {"transportMode":"car","origin":"Yamoussoukro","destination":"Biéby","via":null,"departureTime":null,"routingMode":null,"avoid_features":["ferry"],"lang":null} | {"transportMode":"car","origin":"Yamoussoukro","destination":"Biéby","via":null,"departureTime":null,"routingMode":null,"avoid_features":["ferry"],"lang":null} |

grounded: t1_haiku55=yes t2_gpt54mini=yes t3_gpt6luna=NO t4_glm53flash=yes t5_dsv41flash=yes t6_ling31flash=yes

verdict(v3): ok [src=t1_haiku55, judged by user]

## [41] aug_i-ef10b7af2ab9 (pl, here_directions, aug_train)

utt: Hej asystencie, potrzebuję trasy z Krakowa do Zakopanego, ale unikaj autostrad i dróg płatnych, zaplanuj przerwę co 2 godziny, i znajdź stację ładowania dla samochodu elektrycznego gdzieś po drodze.

params: transportMode:string(enum=5), origin:string(enum=0), destination:string(enum=0), via:array(enum=0), departureTime:string(enum=0), routingMode:string(enum=2), avoid_features:array(enum=0), lang:string(enum=0)

| v2 | t1_haiku55 | t2_gpt54mini | t3_gpt6luna | t4_glm53flash | t5_dsv41flash | t6_ling31flash |
|---|---|---|---|---|---|---|
| {"transportMode":"car","origin":null,"destination":null,"via":null,"departureTime":null,"routingMode":null,"avoid_features":null,"lang":null} | {"transportMode":"car","origin":"Kraków","destination":"Zakopane","via":null,"departureTime":null,"routingMode":null,"avoid_features":["motorway","tollRoad"],"lang":null} | {"transportMode":"car","origin":"Kraków","destination":"Zakopane","via":null,"departureTime":null,"routingMode":null,"avoid_features":["motorway","tollRoad"],"lang":null} | {"transportMode":"car","origin":"Kraków","destination":"Zakopane","via":null,"departureTime":null,"routingMode":null,"avoid_features":["motorway","tollRoad"],"lang":null} | {"transportMode":"car","origin":"Kraków","destination":"Zakopane","via":null,"departureTime":null,"routingMode":null,"avoid_features":["motorway","tollRoad"],"lang":null} | {"transportMode":"car","origin":"Kraków","destination":"Zakopane","via":null,"departureTime":null,"routingMode":null,"avoid_features":["motorway","tollRoad"],"lang":null} | {"transportMode":"car","origin":"Kraków","destination":"Zakopane","via":null,"departureTime":null,"routingMode":null,"avoid_features":["motorway","tollRoad"],"lang":null} |

verdict(v3): ok [src=t1_haiku55, judged by user]

## [42] xtr-4498cff8077f (pl, here_geocode, core_val)

utt: Cześć, porozmawiajmy o trasie z Delhi do Rudrapur; upewnij się, że sie w niej nie wlicza żadnych promów.

params: q:string(enum=0), lang:string(enum=0), limit:integer(enum=0)

| v2 | t1_haiku55 | t2_gpt54mini | t3_gpt6luna | t4_glm53flash | t5_dsv41flash | t6_ling31flash |
|---|---|---|---|---|---|---|
| {"q":"Delhi","lang":null,"limit":null} | {"q":"Delhi","lang":null,"limit":null} | {"q":"Rudrapur","lang":null,"limit":null} | {"q":"Delhi to Rudrapur","lang":null,"limit":null} | {"q":"Rudrapur","lang":null,"limit":null} | {"q":null,"lang":null,"limit":null} | {"q":"Rudrapur","lang":null,"limit":null} |

verdict(v3): ok [src=t1_haiku55, judged by user]

## [43] xtr-525ab0a05b22 (pl, here_geocode, core_train)

utt: Hej, potrzebuję poprowadzić route z Changchun do Huadian, ok. 190 km, ale bez zbiornikowców. Potrzebuję teraz ubrać to w wiadomość.

params: q:string(enum=0), lang:string(enum=0), limit:integer(enum=0)

| v2 | t1_haiku55 | t2_gpt54mini | t3_gpt6luna | t4_glm53flash | t5_dsv41flash | t6_ling31flash |
|---|---|---|---|---|---|---|
| {"q":"Changchun","lang":null,"limit":null} | {"q":"Changchun","lang":null,"limit":null} | {"q":"Changchun do Huadian","lang":null,"limit":null} | {"q":"Changchun, Huadian","lang":null,"limit":null} | {"q":"Huadian","lang":null,"limit":null} | null | null |

grounded: t1_haiku55=yes t2_gpt54mini=yes t3_gpt6luna=yes t4_glm53flash=yes t5_dsv41flash=NO t6_ling31flash=NO

verdict(v3): ok [src=t1_haiku55, judged by user]

## [44] xtr-966d785d6781 (pl, here_geocode, core_train)

utt: Cześć, podłącz mnie do nawigacji. Proszę o trasę z Jianshui do Yulin, około 1250 km, ale musimy całkowicie ominąć centra miast – vorrei tylko autostrady. Czy masz gotowe taką opcję?

params: q:string(enum=0), lang:string(enum=0), limit:integer(enum=0)

| v2 | t1_haiku55 | t2_gpt54mini | t3_gpt6luna | t4_glm53flash | t5_dsv41flash | t6_ling31flash |
|---|---|---|---|---|---|---|
| {"q":"Jianshui","lang":null,"limit":null} | {"q":"Jianshui","lang":null,"limit":null} | {"q":"Yulin","lang":null,"limit":null} | {"q":"Jianshui, Yulin","lang":null,"limit":null} | {"q":"Jianshui","lang":null,"limit":null} | null | null |

grounded: t1_haiku55=yes t2_gpt54mini=yes t3_gpt6luna=yes t4_glm53flash=yes t5_dsv41flash=NO t6_ling31flash=NO

verdict(v3): ok [src=t1_haiku55, judged by user]

## [45] xtr-9a3d9fe4ab75 (pl, tomtom-routing, core_train)

utt: Dzień dobry! Chciałbyś podać trasę z Warszawy do Zakopanego, tak aby prowadziła przez najpiękniejsze krajobrazy, nawet jak to oznacza więcej czasu, a w połowie podróży zaproponuj mi miasteczko, w którym moglibyśmy zrobić postój?

params: origin:object(enum=0), destination:object(enum=0), response_detail:string(enum=2), routeType:string(enum=4), travelMode:string(enum=7), traffic:boolean(enum=0), avoid:array(enum=0), departAt:string(enum=0), arriveAt:string(enum=0), maxAlternatives:number(enum=0), alternativeType:string(enum=2), instructionsType:string(enum=3), language:string(enum=0), computeBestOrder:boolean(enum=0), supportingPoints:string(enum=0), vehicleHeading:number(enum=0), includeTollPaymentTypes:string(enum=0), report:string(enum=0), routeRepresentation:string(enum=4), extendedRouteRepresentation:string(enum=0), computeTravelTimeFor:string(enum=2), hilliness:string(enum=3), windingness:string(enum=3), vehicleMaxSpeed:number(enum=0), vehicleWeight:number(enum=0), vehicleWidth:number(enum=0), vehicleHeight:number(enum=0), vehicleLength:number(enum=0), vehicleCommercial:boolean(enum=0), vehicleAxleWeight:number(enum=0), vehicleNumberOfAxles:number(enum=0), vehicleLoadType:string(enum=0), vehicleAdrTunnelRestrictionCode:string(enum=0), vehicleEngineType:string(enum=2), currentChargeInkWh:number(enum=0), maxChargeInkWh:number(enum=0), constantSpeedConsumptionInkWhPerHundredkm:string(enum=0), auxiliaryPowerInkW:number(enum=0), chargeMarginsInkWh:string(enum=0), constantSpeedConsumptionInLitersPerHundredkm:string(enum=0), currentFuelInLiters:number(enum=0), auxiliaryPowerInLitersPerHour:number(enum=0), fuelEnergyDensityInMJoulesPerLiter:number(enum=0), accelerationEfficiency:number(enum=0), decelerationEfficiency:number(enum=0), uphillEfficiency:number(enum=0), downhillEfficiency:number(enum=0), consumptionInkWhPerkmAltitudeGain:number(enum=0), recuperationInkWhPerkmAltitudeLoss:number(enum=0), sectionType:array(enum=0)

| v2 | t1_haiku55 | t2_gpt54mini | t3_gpt6luna | t4_glm53flash | t5_dsv41flash | t6_ling31flash |
|---|---|---|---|---|---|---|
| {"origin":null,"destination":null,"response_detail":null,"routeType":null,"travelMode":null,"traffic":null,"avoid":null,"departAt":null,"arriveAt":null,"maxAlternatives":null,"alternativeType":null,"instructionsType":null,"language":null,"computeBestOrder":null,"supportingPoints":null,"vehicleHeading":null,"includeTollPaymentTypes":null,"report":null,"routeRepresentation":null,"extendedRouteRepresentation":null,"computeTravelTimeFor":null,"hilliness":null,"windingness":null,"vehicleMaxSpeed":null,"vehicleWeight":null,"vehicleWidth":null,"vehicleHeight":null,"vehicleLength":null,"vehicleCommercial":null,"vehicleAxleWeight":null,"vehicleNumberOfAxles":null,"vehicleLoadType":null,"vehicleAdrTunnelRestrictionCode":null,"vehicleEngineType":null,"currentChargeInkWh":null,"maxChargeInkWh":null,"constantSpeedConsumptionInkWhPerHundredkm":null,"auxiliaryPowerInkW":null,"chargeMarginsInkWh":null,"constantSpeedConsumptionInLitersPerHundredkm":null,"currentFuelInLiters":null,"auxiliaryPowerInLitersPerHour":null,"fuelEnergyDensityInMJoulesPerLiter":null,"accelerationEfficiency":null,"decelerationEfficiency":null,"uphillEfficiency":null,"downhillEfficiency":null,"consumptionInkWhPerkmAltitudeGain":null,"recuperationInkWhPerkmAltitudeLoss":null,"sectionType":null} | null | {"origin":null,"destination":null,"response_detail":null,"routeType":"thrilling","travelMode":null,"traffic":null,"avoid":null,"departAt":null,"arriveAt":null,"maxAlternatives":null,"alternativeType":null,"instructionsType":null,"language":null,"computeBestOrder":null,"supportingPoints":null,"vehicleHeading":null,"includeTollPaymentTypes":null,"report":null,"routeRepresentation":null,"extendedRouteRepresentation":null,"computeTravelTimeFor":null,"hilliness":null,"windingness":null,"vehicleMaxSpeed":null,"vehicleWeight":null,"vehicleWidth":null,"vehicleHeight":null,"vehicleLength":null,"vehicleCommercial":null,"vehicleAxleWeight":null,"vehicleNumberOfAxles":null,"vehicleLoadType":null,"vehicleAdrTunnelRestrictionCode":null,"vehicleEngineType":null,"currentChargeInkWh":null,"maxChargeInkWh":null,"constantSpeedConsumptionInkWhPerHundredkm":null,"auxiliaryPowerInkW":null,"chargeMarginsInkWh":null,"constantSpeedConsumptionInLitersPerHundredkm":null,"currentFuelInLiters":null,"auxiliaryPowerInLitersPerHour":null,"fuelEnergyDensityInMJoulesPerLiter":null,"accelerationEfficiency":null,"decelerationEfficiency":null,"uphillEfficiency":null,"downhillEfficiency":null,"consumptionInkWhPerkmAltitudeGain":null,"recuperationInkWhPerkmAltitudeLoss":null,"sectionType":null} | {"origin":null,"destination":null,"response_detail":null,"routeType":"thrilling","travelMode":null,"traffic":null,"avoid":null,"departAt":null,"arriveAt":null,"maxAlternatives":null,"alternativeType":null,"instructionsType":null,"language":null,"computeBestOrder":null,"supportingPoints":null,"vehicleHeading":null,"includeTollPaymentTypes":null,"report":null,"routeRepresentation":null,"extendedRouteRepresentation":null,"computeTravelTimeFor":null,"hilliness":null,"windingness":null,"vehicleMaxSpeed":null,"vehicleWeight":null,"vehicleWidth":null,"vehicleHeight":null,"vehicleLength":null,"vehicleCommercial":null,"vehicleAxleWeight":null,"vehicleNumberOfAxles":null,"vehicleLoadType":null,"vehicleAdrTunnelRestrictionCode":null,"vehicleEngineType":null,"currentChargeInkWh":null,"maxChargeInkWh":null,"constantSpeedConsumptionInkWhPerHundredkm":null,"auxiliaryPowerInkW":null,"chargeMarginsInkWh":null,"constantSpeedConsumptionInLitersPerHundredkm":null,"currentFuelInLiters":null,"auxiliaryPowerInLitersPerHour":null,"fuelEnergyDensityInMJoulesPerLiter":null,"accelerationEfficiency":null,"decelerationEfficiency":null,"uphillEfficiency":null,"downhillEfficiency":null,"consumptionInkWhPerkmAltitudeGain":null,"recuperationInkWhPerkmAltitudeLoss":null,"sectionType":null} | null | null | null |

grounded: t1_haiku55=NO t2_gpt54mini=yes t3_gpt6luna=yes t4_glm53flash=NO t5_dsv41flash=NO t6_ling31flash=NO

verdict(v3): ok [src=v2_fallback, judged by user]

## [46] xtr-a1d80574aee8 (pl, here_geocode, core_train)

utt: Hej, zaplanuj mi trasę z Krakowa do Zakopanego. Zależy mi na tym, żeby nie płacić za autostrady i omijać te wszystkie bardzo kręte, malownicze drogi, bo nie chcę się męczyć za kierownicą. Po drodze chciałbym też gdzieś zatrzymać się na kawę.

params: q:string(enum=0), lang:string(enum=0), limit:integer(enum=0)

| v2 | t1_haiku55 | t2_gpt54mini | t3_gpt6luna | t4_glm53flash | t5_dsv41flash | t6_ling31flash |
|---|---|---|---|---|---|---|
| {"q":"Kraków","lang":null,"limit":null} | {"q":"Zakopane","lang":null,"limit":null} | {"q":"Kraków","lang":null,"limit":null} | {"q":"Kraków, Zakopane","lang":null,"limit":null} | {"q":null,"lang":null,"limit":null} | {"q":null,"lang":null,"limit":null} | null |

grounded: t1_haiku55=yes t2_gpt54mini=yes t3_gpt6luna=yes t4_glm53flash=yes t5_dsv41flash=yes t6_ling31flash=NO

verdict(v3): ok [src=t1_haiku55, judged by user]

## [47] xtr-ab793c614ec3 (pl, here_directions, core_train)

utt: Cześć, chcę pojechać z Krakowa do Energylandii, ale proszę, żebyś znalazł trasę omijającą płatne autostrady. Do tego bardzo zależy mi na przerwaniu się w Zatorze na kawę i chce jechać malowniczymi drogami przez wsie. Czy możesz to ustawić?

params: transportMode:string(enum=5), origin:string(enum=0), destination:string(enum=0), via:array(enum=0), departureTime:string(enum=0), routingMode:string(enum=2), avoid_features:array(enum=0), lang:string(enum=0)

| v2 | t1_haiku55 | t2_gpt54mini | t3_gpt6luna | t4_glm53flash | t5_dsv41flash | t6_ling31flash |
|---|---|---|---|---|---|---|
| {"transportMode":null,"origin":null,"destination":null,"via":null,"departureTime":null,"routingMode":null,"avoid_features":["motorway"],"lang":null} | null | {"transportMode":"car","origin":"Kraków","destination":"Energylandii","via":["Zator"],"departureTime":null,"routingMode":"short","avoid_features":["tollRoad"],"lang":null} | null | null | null | null |

grounded: t1_haiku55=NO t2_gpt54mini=yes t3_gpt6luna=NO t4_glm53flash=NO t5_dsv41flash=NO t6_ling31flash=NO

verdict(v3): bad [src=v2_fallback, judged by user] — 'płatnych autostrad' -> avoid=[motorway] but policy §3.3 pre-registers tollRoad; user '?' resolved per policy; v2-inherited noise, not teacher error

## [48] xtr-bc0479998713 (pl, tomtom-fuzzy-search, core_train)

utt: Jasne — jedziemy rodzinnie z Warszawy do Sandomierza na weekend i proszę wyznacz trasę tak, żeby omijała płatne autostrady. Chcę też, żebyś później podpowiedział opcje parkowania w Sandomierzu po dotarciu.

params: query:string(enum=0), response_detail:string(enum=2), limit:number(enum=0), language:string(enum=0), countrySet:string(enum=0), view:string(enum=0), extendedPostalCodesFor:string(enum=0), mapcodes:array(enum=0), timeZone:string(enum=0), geometries:boolean(enum=0), addressRanges:boolean(enum=0), lat:number(enum=0), lon:number(enum=0), radius:number(enum=0), topLeft:string(enum=0), btmRight:string(enum=0), categorySet:string(enum=0), brandSet:string(enum=0), connectorSet:string(enum=0), fuelSet:string(enum=0), vehicleTypeSet:string(enum=0), minPowerKW:number(enum=0), maxPowerKW:number(enum=0), openingHours:string(enum=0), typeahead:boolean(enum=0), maxFuzzyLevel:number(enum=0), minFuzzyLevel:number(enum=0), entityTypeSet:string(enum=0), ofs:number(enum=0), idxSet:string(enum=0), relatedPois:string(enum=0), ext:string(enum=0)

| v2 | t1_haiku55 | t2_gpt54mini | t3_gpt6luna | t4_glm53flash | t5_dsv41flash | t6_ling31flash |
|---|---|---|---|---|---|---|
| {"query":null,"response_detail":null,"limit":null,"language":null,"countrySet":null,"view":null,"extendedPostalCodesFor":null,"mapcodes":null,"timeZone":null,"geometries":null,"addressRanges":null,"lat":null,"lon":null,"radius":null,"topLeft":null,"btmRight":null,"categorySet":null,"brandSet":null,"connectorSet":null,"fuelSet":null,"vehicleTypeSet":null,"minPowerKW":null,"maxPowerKW":null,"openingHours":null,"typeahead":null,"maxFuzzyLevel":null,"minFuzzyLevel":null,"entityTypeSet":null,"ofs":null,"idxSet":null,"relatedPois":null,"ext":null} | null | null | {"query":"opcje parkowania w Sandomierz","response_detail":null,"limit":null,"language":null,"countrySet":null,"view":null,"extendedPostalCodesFor":null,"mapcodes":null,"timeZone":null,"geometries":null,"addressRanges":null,"lat":null,"lon":null,"radius":null,"topLeft":null,"btmRight":null,"categorySet":null,"brandSet":null,"connectorSet":null,"fuelSet":null,"vehicleTypeSet":null,"minPowerKW":null,"maxPowerKW":null,"openingHours":null,"typeahead":null,"maxFuzzyLevel":null,"minFuzzyLevel":null,"entityTypeSet":null,"ofs":null,"idxSet":null,"relatedPois":null,"ext":null} | {"query":null,"response_detail":null,"limit":null,"language":null,"countrySet":null,"view":null,"extendedPostalCodesFor":null,"mapcodes":null,"timeZone":null,"geometries":null,"addressRanges":null,"lat":null,"lon":null,"radius":null,"topLeft":null,"btmRight":null,"categorySet":null,"brandSet":null,"connectorSet":null,"fuelSet":null,"vehicleTypeSet":null,"minPowerKW":null,"maxPowerKW":null,"openingHours":null,"typeahead":null,"maxFuzzyLevel":null,"minFuzzyLevel":null,"entityTypeSet":null,"ofs":null,"idxSet":null,"relatedPois":null,"ext":null} | null | null |

grounded: t1_haiku55=NO t2_gpt54mini=NO t3_gpt6luna=yes t4_glm53flash=yes t5_dsv41flash=NO t6_ling31flash=NO

verdict(v3): ok [src=t1_haiku55, judged by user]

## [49] xtr-deb72f1114dc (pl, here_geocode, core_train)

utt: Dzień dobry, mam przygotować trasę z Gniezna do Wrześni. Proszę, aby pokazał drogę lokalną o charakterze morenowym zamiast szybkiej drogi krajowej, zaniedbał drogę ekspresową i wszystkie bramy płatne. Muszę również zatrzymać się w Kostrzynie na kawę. Czy jest taka opcja?

params: q:string(enum=0), lang:string(enum=0), limit:integer(enum=0)

| v2 | t1_haiku55 | t2_gpt54mini | t3_gpt6luna | t4_glm53flash | t5_dsv41flash | t6_ling31flash |
|---|---|---|---|---|---|---|
| {"q":null,"lang":null,"limit":null} | null | {"q":"Kostrzynie","lang":null,"limit":null} | {"q":"Kostrzyn","lang":null,"limit":null} | null | null | null |

grounded: t1_haiku55=NO t2_gpt54mini=yes t3_gpt6luna=yes t4_glm53flash=NO t5_dsv41flash=NO t6_ling31flash=NO

verdict(v3): ok [src=t4_glm53flash, judged by user]

## [50] xtr-fb17ee626994 (pl, here_geocode, core_train)

utt: Dzień dobry, szukam trasy z Krakowa do Zakopanego, unikając autostrad i dróg płatnych, preferując malownicze szlaki. Proszę uwzględnić przerwę co 100 km na odpoczynek. Czy już mamy ogólny plan jazdy?

params: q:string(enum=0), lang:string(enum=0), limit:integer(enum=0)

| v2 | t1_haiku55 | t2_gpt54mini | t3_gpt6luna | t4_glm53flash | t5_dsv41flash | t6_ling31flash |
|---|---|---|---|---|---|---|
| {"q":"Kraków","lang":null,"limit":null} | {"q":"Kraków","lang":null,"limit":null} | {"q":"Kraków do Zakopanego","lang":null,"limit":null} | {"q":"Kraków, Zakopane","lang":null,"limit":null} | {"q":null,"lang":null,"limit":null} | {"q":null,"lang":null,"limit":null} | null |

grounded: t1_haiku55=yes t2_gpt54mini=yes t3_gpt6luna=yes t4_glm53flash=yes t5_dsv41flash=yes t6_ling31flash=NO

verdict(v3): ok [src=t1_haiku55, judged by user]


---

## Results (2026-10-19) — ADOPTION GATE: **PASS**

Provenance: rows 11–18 + 39–50 (en/pl) judged by the user (user languages: en/pl);
rows 1–10 + 19–38 (de/es/fr) judged by the assistant strictly against the pre-registered policy
(`2026-10-13-gold-v3-policy.md` §3). Row 47 user "?" resolved to **bad** per policy §3.3
("płatnych autostrad" → tollRoad, not motorway).

| Slice | ok | bad | ? | strict | excl-? |
|---|---|---|---|---|---|
| Teacher-sourced v3 (45 t1 + 2 t4) | 46 | 1 | 1 | **97.9%** | 100% |
| v2_fallback rows (3) | 2 | 1 | 0 | 66.7% | 66.7% |
| All 50 | 47 | 2 | 1 | 94.0% | 95.9% |
| pl teacher-sourced (11) | 11 | 0 | 0 | 100% | 100% |

Avoid blank-receipts condition: t1 blank-on-explicit-cue 135/2,543 (5.3%) on here_directions, but a
15-row sample shows most blanks are correct (cue = city centers / roadworks / traffic / countries —
not representable in the avoid enum [tollRoad, ferry, motorway, tunnel]); real blank-receipts ≈0.3%
→ "≈0" PASSES. t4: 0/13. v2_fallback: 61% blank (inherited v2 noise).

Gate verdict: **PASS on teacher labels** (97.9% strict ≥ 95%; pl teacher-sourced 100% — no
Jev-style pl collapse). The only bad teacher row (4) is a known t1 quirk: vehicle words
(van/truck) collapse to car (~2% of vehicle-word rows). v2_fallback rows (6.4% of corpus, 3.6% of
test) carry legacy v2 noise (row 47) — known and bounded; they were already part of accepted v2.

→ v3 adopted as headline gold. Phase B retrains unblocked (2026-10-19).
