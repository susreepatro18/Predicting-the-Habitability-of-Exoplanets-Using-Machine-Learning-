window.addEventListener("DOMContentLoaded", loadRanking);

function loadRanking() {
  fetch("/ranking")
    .then(res => res.json())
    .then(data => {
      const table = document.getElementById("rankingTable");
      table.innerHTML = "";

      data.forEach((p, i) => {
        table.innerHTML += `
          <tr>
            <td>${i + 1}</td>
            <td>${p.pl_name}</td>
            <td>${p.hostname}</td>
            <td>${Number(p.habitability_score).toFixed(3)}</td>
          </tr>
        `;
      });
    });
}

function predict() {
  fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      planet_radius: Number(radius.value),
      planet_mass: Number(mass.value),
      surface_temperature: Number(temp.value)
    })
  })
  .then(res => res.json())
  .then(d => {
    document.getElementById("result").innerText =
      `Habitability: ${d.habitability_class} | Score: ${d.habitability_score}`;
  });
}
