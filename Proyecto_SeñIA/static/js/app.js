
let historialGestos = [];
let ultimaEtiqueta = "..."; // Esta variable global guarda el **último gesto detectado** para evitar repetir el mismo audio si 
// no ha cambiado

function iniciarDeteccion() {
    // Reproduce el audio de "hola" al iniciar
    document.getElementById("audio-presentacion").play().catch(() => {});

    setInterval(() => {
        fetch('/get_gesto') //solicitud http al endpoint
            .then(response => response.json()) //la respuesta debe venir en un json(sera el gesto detectado)
            .then(data => { //la respues que recibio(json) la extrae y guarda en la variable gesto
                const gesto = data.gesto;
                if (gesto !== ultimaEtiqueta) {
                     let utterance = new SpeechSynthesisUtterance(gesto);
                     utterance.lang = "es-AR";  // Español
                     speechSynthesis.speak(utterance);
                     ultimaEtiqueta = gesto;
                     const timestamp = new Date().toLocaleTimeString();
                     historialGestos.push({ gesto, timestamp });

                    
                     const li = document.createElement("li");
                     li.textContent = `[${timestamp}] ${gesto}`;
                     document.getElementById("listaHistorial").appendChild(li);
                   
                }
            });
    }, 1000);
}


function exportarPDF() {
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF();

    doc.setFontSize(16);
    doc.text("Resumen de señas detectadas", 10, 10);

    let y = 20;
    historialGestos.forEach((item, index) => {
        doc.text(`${index + 1}. ${item.timestamp} - ${item.gesto}`, 10, y);
        y += 10;
    });

    doc.save("resumen_señas.pdf");
}

function limpiarHistorial() {
    historialGestos = [];
    document.getElementById("listaHistorial").innerHTML = "";
}