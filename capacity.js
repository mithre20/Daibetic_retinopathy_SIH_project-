/* ============================================================
   Capacity Planning — M/M/c queueing model
   JavaScript port of capacityPlanningAnalysis.m
   ============================================================ */

class CapacityPlanner {
    constructor(params = {}) {
        this.params = {
            annualPatients: params.annualPatients || 100000,
            screeningDaysPerYear: params.screeningDaysPerYear || 300,
            hoursPerDay: params.hoursPerDay || 8,
            imageSizeMB: params.imageSizeMB || 2.5,
            uplinkMbps: params.uplinkMbps || 4,
            aiProcessingSecPerImage: params.aiProcessingSecPerImage || 8,
            reviewMinPerCase: params.reviewMinPerCase || 0.5,
            referableFraction: params.referableFraction || 0.18,
            targetUtilization: params.targetUtilization || 0.75
        };
    }

    /** M/M/c server sizing with Erlang-C wait time */
    sizeServers(lambda, mu, targetUtil) {
        if (lambda <= 0) return { c: 0, rho: 0, Wq: 0 };

        const c = Math.max(1, Math.ceil(lambda / (mu * targetUtil)));
        const a = lambda / mu; // offered load (Erlangs)
        const rho = a / c;

        // Erlang-C probability of waiting
        let sumTerm = 0;
        for (let k = 0; k < c; k++) {
            sumTerm += Math.pow(a, k) / this._factorial(k);
        }
        const lastTerm = (Math.pow(a, c) / this._factorial(c)) * (1 / (1 - rho));
        const Pwait = lastTerm / (sumTerm + lastTerm);
        let Wq = Pwait / (c * mu - lambda);
        if (!isFinite(Wq) || Wq < 0) Wq = 0;

        return { c, rho, Wq };
    }

    _factorial(n) {
        if (n <= 1) return 1;
        let f = 1;
        for (let i = 2; i <= n; i++) f *= i;
        return f;
    }

    /** Run the full capacity analysis */
    analyze() {
        const o = this.params;

        // Arrival rate
        const totalScreeningHours = o.screeningDaysPerYear * o.hoursPerDay;
        const lambdaPerHour = o.annualPatients / totalScreeningHours;
        const lambdaPerSec = lambdaPerHour / 3600;

        // 1. Upload / bandwidth stage
        const uploadSecPerImage = (o.imageSizeMB * 8) / o.uplinkMbps;
        const muUpload = 1 / uploadSecPerImage;
        const upload = this.sizeServers(lambdaPerSec, muUpload, o.targetUtilization);

        // 2. AI processing stage
        const muAI = 1 / o.aiProcessingSecPerImage;
        const ai = this.sizeServers(lambdaPerSec, muAI, o.targetUtilization);

        // 3. Human review stage (only referable/flagged cases)
        const lambdaReview = lambdaPerSec * o.referableFraction;
        const muReview = 1 / (o.reviewMinPerCase * 60);
        const review = this.sizeServers(lambdaReview, muReview, o.targetUtilization);
        const reviewerHoursPerDay = review.c * o.hoursPerDay;

        // Sensitivity sweep
        const reviewTimes = [0.25, 0.5, 0.75, 1.0];
        const referralFracs = [0.10, 0.18, 0.25];
        const sweep = [];
        for (const rt of reviewTimes) {
            const row = [];
            for (const rf of referralFracs) {
                const lam = lambdaPerSec * rf;
                const mu = 1 / (rt * 60);
                const result = this.sizeServers(lam, mu, o.targetUtilization);
                row.push(result.c);
            }
            sweep.push(row);
        }

        return {
            lambdaPerHour,
            lambdaPerSec,
            uploadSecPerImage,
            upload: { ...upload, serviceTime: uploadSecPerImage },
            ai: { ...ai, serviceTime: o.aiProcessingSecPerImage },
            review: { ...review, serviceTime: o.reviewMinPerCase * 60, reviewerHoursPerDay },
            sensitivity: {
                reviewTimes,
                referralFracs,
                requiredServers: sweep
            }
        };
    }
}

/** Render capacity results into the DOM */
function renderCapacityResults(results) {
    const stageCards = document.getElementById('stage-cards');
    stageCards.innerHTML = `
        <div class="stage-card upload">
            <div class="stage-icon">📡</div>
            <div class="stage-title">Upload Channels</div>
            <div class="stage-value">${results.upload.c}</div>
            <div class="stage-unit">parallel gateways</div>
            <div class="stage-meta">
                ${results.uploadSecPerImage.toFixed(1)}s per case<br>
                ${(results.upload.rho * 100).toFixed(0)}% utilization<br>
                ${results.upload.Wq.toFixed(1)}s avg wait
            </div>
        </div>
        <div class="stage-card ai">
            <div class="stage-icon">🤖</div>
            <div class="stage-title">AI Workers</div>
            <div class="stage-value">${results.ai.c}</div>
            <div class="stage-unit">CPU/GPU instances</div>
            <div class="stage-meta">
                ${results.ai.serviceTime.toFixed(1)}s per case<br>
                ${(results.ai.rho * 100).toFixed(0)}% utilization<br>
                ${results.ai.Wq.toFixed(1)}s avg wait
            </div>
        </div>
        <div class="stage-card review">
            <div class="stage-icon">👨‍⚕️</div>
            <div class="stage-title">Reviewers</div>
            <div class="stage-value">${results.review.c}</div>
            <div class="stage-unit">ophthalmologists online</div>
            <div class="stage-meta">
                ${results.review.serviceTime.toFixed(0)}s per case<br>
                ${(results.review.rho * 100).toFixed(0)}% utilization<br>
                ${results.review.Wq.toFixed(1)}s avg wait<br>
                ${results.review.reviewerHoursPerDay}h/day needed
            </div>
        </div>
    `;

    // Sensitivity table
    const sens = results.sensitivity;
    const table = document.getElementById('sensitivity-table');
    let html = '<thead><tr><th>Review Time</th>';
    for (const rf of sens.referralFracs) html += `<th>${(rf * 100).toFixed(0)}% Referral</th>`;
    html += '</tr></thead><tbody>';
    for (let i = 0; i < sens.reviewTimes.length; i++) {
        html += `<tr><td>${sens.reviewTimes[i].toFixed(2)} min</td>`;
        for (let j = 0; j < sens.referralFracs.length; j++) {
            html += `<td>${sens.requiredServers[i][j]}</td>`;
        }
        html += '</tr>';
    }
    html += '</tbody>';
    table.innerHTML = html;

    // Workflow diagram
    const wf = document.getElementById('workflow-diagram');
    wf.innerHTML = `
        <div class="wf-node source">
            <div class="wf-icon">🏥</div>
            <div class="wf-label">Patient Arrivals</div>
            <div class="wf-detail">${results.lambdaPerHour.toFixed(1)}/hr</div>
        </div>
        <div class="wf-arrow">→</div>
        <div class="wf-node upload">
            <div class="wf-icon">📡</div>
            <div class="wf-label">Upload</div>
            <div class="wf-detail">${results.upload.c} channels</div>
        </div>
        <div class="wf-arrow">→</div>
        <div class="wf-node ai">
            <div class="wf-icon">🤖</div>
            <div class="wf-label">AI Processing</div>
            <div class="wf-detail">${results.ai.c} workers</div>
        </div>
        <div class="wf-arrow">→</div>
        <div class="wf-node split">
            <div class="wf-icon">🔀</div>
            <div class="wf-label">Referral Split</div>
            <div class="wf-detail">${(parseFloat(document.getElementById('cap-referable').value) * 100).toFixed(0)}% referable</div>
        </div>
        <div class="wf-arrow">→</div>
        <div class="wf-node review">
            <div class="wf-icon">👨‍⚕️</div>
            <div class="wf-label">Human Review</div>
            <div class="wf-detail">${results.review.c} reviewers</div>
        </div>
        <div class="wf-arrow">→</div>
        <div class="wf-node sink">
            <div class="wf-icon">✅</div>
            <div class="wf-label">Cleared / Referred</div>
            <div class="wf-detail">Report generated</div>
        </div>
    `;

    document.getElementById('capacity-results').style.display = 'block';
}

window.CapacityPlanner = CapacityPlanner;
window.renderCapacityResults = renderCapacityResults;
