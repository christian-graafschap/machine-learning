const defaultData = {
    longitude: -122.23,
    latitude: 37.88,
    housing_median_age: 41,
    total_rooms: 880,
    total_bedrooms: 129,
    population: 322,
    households: 126,
    median_income: 8.3252,
    ocean_proximity: "NEAR BAY"
};

// 🔥 vul inputs bij laden
window.onload = () => {
    document.getElementById("longitude").value = defaultData.longitude;
    document.getElementById("latitude").value = defaultData.latitude;
    document.getElementById("age").value = defaultData.housing_median_age;
    document.getElementById("rooms").value = defaultData.total_rooms;
    document.getElementById("bedrooms").value = defaultData.total_bedrooms;
    document.getElementById("population").value = defaultData.population;
    document.getElementById("households").value = defaultData.households;
    document.getElementById("income").value = defaultData.median_income;
    document.getElementById("ocean").value = defaultData.ocean_proximity;
};

async function predict() {

    const data = {
        longitude: parseFloat(document.getElementById("longitude").value),
        latitude: parseFloat(document.getElementById("latitude").value),
        housing_median_age: parseFloat(document.getElementById("age").value),
        total_rooms: parseFloat(document.getElementById("rooms").value),
        total_bedrooms: parseFloat(document.getElementById("bedrooms").value),
        population: parseFloat(document.getElementById("population").value),
        households: parseFloat(document.getElementById("households").value),
        median_income: parseFloat(document.getElementById("income").value),
        ocean_proximity: document.getElementById("ocean").value
    };

    try {
        const response = await fetch("http://localhost:8000/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        document.getElementById("result").innerText =
            "🏠 Predicted price: $" + result.prediction;

    } catch (error) {
        document.getElementById("result").innerText =
            "Error calling API";
        console.error(error);
    }
}