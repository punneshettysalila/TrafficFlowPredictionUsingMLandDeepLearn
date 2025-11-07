// ---- Traffic Prediction Section ----

// Chart.js setup
const ctx = document.getElementById('trafficChart').getContext('2d');
let trafficChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: [],
        datasets: [{
            label: 'Predicted Traffic Flow',
            backgroundColor: 'rgba(8,217,214,0.2)',
            borderColor: '#08d9d6',
            data: [],
            tension: 0.3
        }]
    },
    options: {
        responsive: false,
        plugins: {legend: {labels: {color: '#fff'}}},
        scales: {x: {ticks: {color: '#fff'}}, y: {ticks: {color: '#fff'}}}
    }
});

// Array to collect history of [timestamp, hour, weekday, junction, prediction]
let predictionHistory = [];

document.getElementById('predictForm').addEventListener('submit', function(e) {
    e.preventDefault();
    const hour = document.getElementById('hour').value;
    const weekday = document.getElementById('weekday').value;
    const junction = document.getElementById('junction').value;
    let resultSection = document.getElementById('resultSection');
    resultSection.innerHTML = '<div class="spinner"></div>';

    if (hour < 0 || weekday < 0 || junction < 0) {
        resultSection.innerHTML = `<p style="color:red;">Negative values not allowed for prediction.</p>`;
        return;
    }

    fetch('/predict', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            hour: Number(hour),
            weekday: Number(weekday),
            Junction: Number(junction)
        })
    })
    .then(res => res.json())
    .then(res => {
        if (res.error) {
            resultSection.innerHTML = `<p style="color:red;">${res.error}</p>`;
        } else {
            const pred = Number(res.predicted_traffic).toFixed(2);
            resultSection.innerHTML = `<h2>Predicted Traffic Flow:</h2>
            <p style="font-size:2.5rem;color:#08d9d6;text-shadow:0 0 10px #08d9d6;">
                ${pred}
            </p>`;
            let now = new Date().toLocaleTimeString();
            trafficChart.data.labels.push(now);
            trafficChart.data.datasets[0].data.push(Number(pred));
            if (trafficChart.data.labels.length > 10) {
                trafficChart.data.labels.shift();
                trafficChart.data.datasets[0].data.shift();
            }
            trafficChart.update();

            // Save prediction in history for download
            predictionHistory.push({
                timestamp: now,
                hour,
                weekday,
                junction,
                predicted_traffic: pred
            });
        }
    })
    .catch(() => {
        resultSection.innerHTML = `<p style="color:red;">Prediction failed. Please try again later.</p>`;
    });
});

// ---- CSV Download Section ----
document.getElementById('downloadPredictions').addEventListener('click', function() {
    if (predictionHistory.length === 0) {
        alert("No predictions to download yet!");
        return;
    }
    let csv = "timestamp,hour,weekday,junction,predicted_traffic\n";
    predictionHistory.forEach(row => {
        csv += `${row.timestamp},${row.hour},${row.weekday},${row.junction},${row.predicted_traffic}\n`;
    });

    const blob = new Blob([csv], { type: 'text/csv' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = "prediction_history.csv";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
});
