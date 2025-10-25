document.addEventListener("DOMContentLoaded", () => {
    const addSymptomButton = document.getElementById("addSymptom");
    const symptomsContainer = document.getElementById("symptomsContainer");

    let symptomCount = 1;

    addSymptomButton.addEventListener("click", () => {
        symptomCount++;
        const label = document.createElement("label");
        label.textContent = `Symptom ${symptomCount}:`;
        label.setAttribute("for", `symptom${symptomCount}`);

        const input = document.createElement("input");
        input.type = "text";
        input.id = `symptom${symptomCount}`;
        input.name = `symptom${symptomCount}`;
        input.placeholder = `Describe symptom ${symptomCount}`;

        symptomsContainer.appendChild(label);
        symptomsContainer.appendChild(input);
    });

    const form = document.getElementById("diseaseForm");
    form.addEventListener("submit", (event) => {
        event.preventDefault();
        const formData = new FormData(form);
        const data = Object.fromEntries(formData.entries());
        console.log("Form Data:", data);

        // Placeholder: Integrate your ML model here
    });
});