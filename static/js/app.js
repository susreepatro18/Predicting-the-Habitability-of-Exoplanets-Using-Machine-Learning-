const API="SECRET123";

function val(id){ return parseFloat(document.getElementById(id).value); }

function predict(){
const data={
planet_name:planet_name.value||"Unknown",
pl_rade:val("pl_rade"),
pl_bmasse:val("pl_bmasse"),
pl_eqt:val("pl_eqt"),
st_teff:val("st_teff"),
st_rad:val("st_rad"),
st_mass:val("st_mass"),
st_lum:val("st_lum"),
sy_dist:val("sy_dist")
};

fetch("/predict",{method:"POST",headers:{
"Content-Type":"application/json","x-api-key":API},
body:JSON.stringify(data)})
.then(r=>r.json())
.then(r=>{
score.innerText=r.habitability_score;
status.innerText=r.habitability_prediction?"🌍 Habitable":"❌ Not Habitable";

fetch("/store",{method:"POST",headers:{
"Content-Type":"application/json","x-api-key":API},
body:JSON.stringify(data)}).then(loadRanking);
});
}

function loadRanking(){
    axios.get("/ranking",{headers:{"x-api-key":API_KEY}})
    .then(res=>{
        let rows="";
        res.data.forEach(r=>{
            rows+=`
            <tr>
                <td><span class="rank-badge">#${r.rank}</span></td>
                <td><b>${r.planet_name}</b></td>
                <td><span class="score-badge">${r.habitability_score.toFixed(3)}</span></td>
                <td>
                    <span class="${r.habitability_score>=0.4 ? 'habitable' : 'not-habitable'}">
                        ${r.habitability_score>=0.4 ? '🌍 Habitable' : '❌ Not Habitable'}
                    </span>
                </td>
            </tr>`;
        });
        document.getElementById("ranking").innerHTML=rows;
    });
}

