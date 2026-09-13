from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import psycopg
import joblib
import requests


app = FastAPI()


# =====================================================
# LOAD AI MODEL
# =====================================================

risk_model = joblib.load("risk_model.pkl")


# =====================================================
# DATABASE CONFIGURATION
# =====================================================

DB_CONFIG = {
    "dbname": "Xlambi",
    "user": "postgres",
    "password": 936208,
    "host": "localhost",
    "port": 5432
}


# =====================================================
# DATA MODELS
# =====================================================

class GPSData(BaseModel):
    vehicle_id: str
    latitude: float
    longitude: float
    accuracy: float | None = None
    speed: float | None = None


class RoutePoint(BaseModel):
    latitude: float
    longitude: float


class RouteData(BaseModel):
    points: list[RoutePoint]


# =====================================================
# HOME
# =====================================================

@app.get("/")
def home():

    return {
        "message": "NER-RouteAI GPS Backend is running"
    }


# =====================================================
# GPS PAGE
# =====================================================

@app.get("/gps")
def gps_page():

    return FileResponse("gps.html")


# =====================================================
# MAP PAGE
# =====================================================

@app.get("/map")
def map_page():

    return FileResponse("map.html")


# =====================================================
# CURRENT WEATHER
# =====================================================

@app.get("/api/weather/current")
def get_current_weather(
    latitude: float,
    longitude: float
):

    try:

        url = "https://api.open-meteo.com/v1/forecast"

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": (
                "temperature_2m,"
                "precipitation,"
                "rain,"
                "weather_code,"
                "wind_speed_10m"
            ),
            "timezone": "auto"
        }

        response = requests.get(
            url,
            params=params,
            timeout=10
        )

        response.raise_for_status()

        weather = response.json()

        current = weather["current"]

        return {

            "success": True,

            "temperature":
                current["temperature_2m"],

            "precipitation":
                current["precipitation"],

            "rain":
                current["rain"],

            "weather_code":
                current["weather_code"],

            "wind_speed":
                current["wind_speed_10m"]

        }

    except Exception as e:

        print(
            "WEATHER ERROR:",
            str(e)
        )

        return {

            "success": False,

            "message": str(e)

        }


# =====================================================
# GET LATEST GPS
# =====================================================

@app.get("/api/gps/latest")
def latest_gps():

    try:

        conn = psycopg.connect(
            **DB_CONFIG
        )

        with conn.cursor() as cursor:

            cursor.execute(
                """
                SELECT
                    vehicle_id,
                    latitude,
                    longitude,
                    accuracy,
                    speed,
                    recorded_at
                FROM vehicle_locations
                ORDER BY recorded_at DESC
                LIMIT 1
                """
            )

            row = cursor.fetchone()

        conn.close()


        if row is None:

            return {

                "success": False,

                "message":
                    "No GPS data available"

            }


        return {

            "success": True,

            "vehicle_id": row[0],

            "latitude": row[1],

            "longitude": row[2],

            "accuracy": row[3],

            "speed": row[4],

            "recorded_at": row[5]

        }


    except Exception as e:

        return {

            "success": False,

            "message": str(e)

        }


# =====================================================
# GPS HISTORY
# =====================================================

@app.get("/api/gps/history")
def gps_history():

    try:

        conn = psycopg.connect(
            **DB_CONFIG
        )

        with conn.cursor() as cursor:

            cursor.execute(
                """
                SELECT
                    latitude,
                    longitude
                FROM vehicle_locations
                WHERE vehicle_id = 'V001'
                ORDER BY recorded_at ASC
                """
            )

            rows = cursor.fetchall()

        conn.close()


        return {

            "success": True,

            "locations": [

                {

                    "latitude": row[0],

                    "longitude": row[1]

                }

                for row in rows

            ]

        }


    except Exception as e:

        return {

            "success": False,

            "message": str(e)

        }


# =====================================================
# UPDATE GPS
# =====================================================

@app.post("/api/gps/update")
def update_gps(data: GPSData):

    try:

        conn = psycopg.connect(
            **DB_CONFIG
        )

        with conn.cursor() as cursor:

            cursor.execute(
                """
                INSERT INTO vehicle_locations
                (
                    vehicle_id,
                    latitude,
                    longitude,
                    accuracy,
                    speed,
                    location
                )
                VALUES
                (
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    ST_SetSRID(
                        ST_MakePoint(
                            %s,
                            %s
                        ),
                        4326
                    )::geography
                )
                """,
                (
                    data.vehicle_id,
                    data.latitude,
                    data.longitude,
                    data.accuracy,
                    data.speed,
                    data.longitude,
                    data.latitude
                )
            )

        conn.commit()

        conn.close()


        print(
            "\n========== GPS SAVED =========="
        )

        print(
            "Vehicle ID:",
            data.vehicle_id
        )

        print(
            "Latitude:",
            data.latitude
        )

        print(
            "Longitude:",
            data.longitude
        )

        print(
            "Accuracy:",
            data.accuracy
        )

        print(
            "Speed:",
            data.speed
        )

        print(
            "===============================\n"
        )


        return {

            "success": True,

            "message":
                "GPS location saved to database",

            "vehicle_id":
                data.vehicle_id

        }


    except Exception as e:

        print(
            "DATABASE ERROR:",
            e
        )

        return {

            "success": False,

            "message":
                "Could not save GPS location",

            "error": str(e)

        }


# =====================================================
# GET RISK ZONES
# =====================================================

@app.get("/api/risk-zones")
def get_risk_zones():

    try:

        conn = psycopg.connect(
            **DB_CONFIG
        )

        with conn.cursor() as cursor:

            cursor.execute(
                """
                SELECT
                    id,
                    name,
                    risk_level,
                    latitude,
                    longitude,
                    radius_meters
                FROM risk_zones
                """
            )

            rows = cursor.fetchall()

        conn.close()


        return {

            "success": True,

            "zones": [

                {

                    "id": row[0],

                    "name": row[1],

                    "risk_level": row[2],

                    "latitude": row[3],

                    "longitude": row[4],

                    "radius_meters": row[5]

                }

                for row in rows

            ]

        }


    except Exception as e:

        return {

            "success": False,

            "message": str(e)

        }


# =====================================================
# AI RISK PREDICTION
# =====================================================

@app.post("/api/ai/predict-risk")
def predict_ai_risk(data: GPSData):

    try:

        # =================================================
        # 1. CHECK RISK ZONE
        # =================================================

        conn = psycopg.connect(
            **DB_CONFIG
        )

        risk_zone = 0

        with conn.cursor() as cursor:

            cursor.execute(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM risk_zones
                    WHERE ST_DWithin(

                        ST_SetSRID(
                            ST_MakePoint(
                                %s,
                                %s
                            ),
                            4326
                        )::geography,

                        ST_SetSRID(
                            ST_MakePoint(
                                longitude,
                                latitude
                            ),
                            4326
                        )::geography,

                        radius_meters

                    )
                )
                """,
                (
                    data.longitude,
                    data.latitude
                )
            )

            risk_zone = (
                1
                if cursor.fetchone()[0]
                else 0
            )

        conn.close()


        # =================================================
        # 2. GET REAL-TIME WEATHER
        # =================================================

        weather_url = (
            "https://api.open-meteo.com/v1/forecast"
        )

        weather_params = {

            "latitude":
                data.latitude,

            "longitude":
                data.longitude,

            "current": (
                "temperature_2m,"
                "precipitation,"
                "rain,"
                "weather_code,"
                "wind_speed_10m"
            ),

            "timezone": "auto"

        }


        weather_response = requests.get(
            weather_url,
            params=weather_params,
            timeout=10
        )

        weather_response.raise_for_status()

        weather_data = (
            weather_response.json()
        )

        current_weather = (
            weather_data["current"]
        )


        temperature = (
            current_weather["temperature_2m"]
        )

        precipitation = (
            current_weather["precipitation"]
        )

        rain = (
            current_weather["rain"]
        )

        weather_code = (
            current_weather["weather_code"]
        )

        wind_speed = (
            current_weather["wind_speed_10m"]
        )


        # =================================================
        # 3. WEATHER RISK
        # =================================================

        # 0 = normal weather
        # 1 = risky weather

        if (
            rain > 0
            or precipitation > 0
        ):

            weather_risk = 1


        elif weather_code in [

            45, 48,

            51, 53, 55,
            56, 57,

            61, 63, 65,
            66, 67,

            71, 73, 75,
            77,

            80, 81, 82,

            85, 86,

            95, 96, 99

        ]:

            weather_risk = 1


        else:

            weather_risk = 0


        # =================================================
        # 4. TRAFFIC ESTIMATION
        # =================================================

        speed_kmh = (
            data.speed * 3.6
            if data.speed
            else 0
        )


        # 0 = Low traffic
        # 1 = Medium traffic
        # 2 = High traffic

        if speed_kmh < 15:

            traffic_level = 2

        elif speed_kmh < 30:

            traffic_level = 1

        else:

            traffic_level = 0


        # =================================================
        # 5. RANDOM FOREST AI PREDICTION
        # =================================================

        prediction = risk_model.predict(

            [[

                speed_kmh,

                data.accuracy
                if data.accuracy
                else 0,

                risk_zone,

                weather_risk,

                traffic_level

            ]]

        )


        risk_score = round(
            float(prediction[0])
        )


        risk_score = max(
            0,
            min(
                100,
                risk_score
            )
        )


        # =================================================
        # 6. RISK LEVEL
        # =================================================

        if risk_score >= 70:

            risk_level = "HIGH"

        elif risk_score >= 40:

            risk_level = "MEDIUM"

        else:

            risk_level = "LOW"


        # =================================================
        # 7. RETURN AI RESULT
        # =================================================

        return {

            "success": True,

            "ai_risk_score":
                risk_score,

            "risk_level":
                risk_level,

            "inside_risk_zone":
                bool(risk_zone),

            "weather": {

                "temperature":
                    temperature,

                "precipitation":
                    precipitation,

                "rain":
                    rain,

                "weather_code":
                    weather_code,

                "wind_speed":
                    wind_speed,

                "weather_risk":
                    weather_risk

            },

            "traffic_level":
                traffic_level

        }


    except Exception as e:

        print(
            "AI RISK ERROR:",
            str(e)
        )

        return {

            "success": False,

            "message": str(e)

        }


# =====================================================
# ROUTE RISK ANALYSIS
# =====================================================

@app.post("/api/route/risk")
def calculate_route_risk(
    data: RouteData
):

    try:

        if len(data.points) < 2:

            return {

                "success": False,

                "message":
                    "Route must contain at least 2 points"

            }


        # =================================================
        # CREATE LINESTRING
        # =================================================

        coordinates = []

        for point in data.points:

            coordinates.append(

                f"{point.longitude} "
                f"{point.latitude}"

            )


        route_wkt = (

            "LINESTRING("
            +
            ", ".join(coordinates)
            +
            ")"

        )


        conn = psycopg.connect(
            **DB_CONFIG
        )


        risk_zones_found = []


        with conn.cursor() as cursor:

            cursor.execute(
                """
                SELECT
                    id,
                    name,
                    risk_level,
                    latitude,
                    longitude,
                    radius_meters

                FROM risk_zones

                WHERE ST_DWithin(

                    ST_GeomFromText(
                        %s,
                        4326
                    )::geography,

                    ST_SetSRID(
                        ST_MakePoint(
                            longitude,
                            latitude
                        ),
                        4326
                    )::geography,

                    radius_meters

                )
                """,
                (route_wkt,)
            )


            rows = cursor.fetchall()


            for row in rows:

                risk_zones_found.append({

                    "id":
                        row[0],

                    "name":
                        row[1],

                    "risk_level":
                        row[2],

                    "latitude":
                        row[3],

                    "longitude":
                        row[4],

                    "radius_meters":
                        row[5]

                })


        conn.close()


        # =================================================
        # DETERMINE ROUTE RISK
        # =================================================

        route_risk = "LOW"


        for zone in risk_zones_found:

            if zone["risk_level"] == "HIGH":

                route_risk = "HIGH"

                break

            elif (
                zone["risk_level"]
                == "MEDIUM"
            ):

                route_risk = "MEDIUM"


        # =================================================
        # RETURN RESULT
        # =================================================

        return {

            "success": True,

            "route_risk":
                route_risk,

            "risk_zones":
                risk_zones_found,

            "risk_zone_count":
                len(risk_zones_found)

        }


    except Exception as e:

        print(
            "ROUTE RISK ERROR:",
            str(e)
        )

        return {

            "success": False,

            "message":
                str(e)

        }