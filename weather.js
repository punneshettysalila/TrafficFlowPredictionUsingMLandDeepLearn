const weatherApiKey = "d6e0b35af268f54e479958d713788ad0"; // Only once!
const citySelector = document.getElementById('citySelector');
const weatherInfo = document.getElementById('weatherInfo');

function fetchWeather(city) {
    fetch(`https://api.openweathermap.org/data/2.5/weather?q=${city}&units=metric&appid=${weatherApiKey}`)
        .then(res => res.json())
        .then(data => {
            if (data.cod === 200) {
                weatherInfo.innerHTML = `
                    <div>
                        <img src="<https://openweathermap.org/img/wn/${data.weather>[0].icon}@4x.png" alt="${data.weather[0].main}" />
                    </div>
                    <div>
                        <h3>${data.name}, ${data.sys.country}</h3>
                        <p>${data.weather[0].main} (${data.weather[0].description})</p>
                        <p>🌡️ Temp: ${Math.round(data.main.temp)} °C</p>
                        <p>Min: ${Math.round(data.main.temp_min)} °C, Max: ${Math.round(data.main.temp_max)} °C</p>
                        <p>Humidity: ${data.main.humidity}% &nbsp; Wind: ${data.wind.speed} m/s</p>
                    </div>
                `;
            } else {
                weatherInfo.innerHTML = `<p style="color:red;">Weather data unavailable (${data.message})</p>`;
            }
        })
        .catch(() => {
            weatherInfo.innerHTML = `<p style="color:red;">Weather data unavailable</p>`;
        });
}

if (citySelector && weatherInfo) {
    fetchWeather(citySelector.value);
    citySelector.addEventListener('change', e => fetchWeather(e.target.value));
}
