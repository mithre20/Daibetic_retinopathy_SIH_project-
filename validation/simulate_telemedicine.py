"""
Telemedicine Screening Workflow Simulation & 100,000+ Patients/Year Sizing Engine
AI DR Screen — Explainable AI for Diabetic Retinopathy Screening in Rural India

Mirrors and executes the exact multi-stage queuing architecture of matlab/simulink/ai_dr_screening.slx:
1. Patient Arrival Stream (Poisson dynamic arrival, >= 100,000 patients/year)
2. Camera Acquisition Stage (M/M/c multi-camera queue)
3. Rural Network Upload & Bandwidth Throttle (0.5 to 50 Mbps)
4. Image Quality Assessment (IQA unusable image gate & on-site recapture loop)
5. Automated AI Processing Stage (Measured hardware throughput: 456.7 ms / image)
6. Ophthalmologist Review Stage (M/M/c Erlang-C human reviewer pool)

Simulates Scenarios A to E, performs resource optimization, and exports summaries.
"""

import os
import sys
import math
import json
import csv
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def solve_mmc_queue(lambda_rate, mu_service, c_servers):
    """
    Solves M/M/c multi-server queue using Erlang-C formula.
    Returns: (c, utilization, avg_wait_seconds, avg_queue_length)
    """
    if lambda_rate <= 0 or mu_service <= 0 or c_servers <= 0:
        return c_servers, 0.0, 0.0, 0.0
        
    c = max(1, int(c_servers))
    a = lambda_rate / mu_service  # Offered load (Erlangs)
    utilization = a / c
    
    if utilization >= 0.999:
        return c, 0.999, 3600.0, 100.0
        
    # Erlang-C formula
    sum_terms = sum((a**k) / math.factorial(k) for k in range(c))
    last_term = ((a**c) / math.factorial(c)) * (1.0 / (1.0 - utilization))
    p_wait = last_term / (sum_terms + last_term)
    
    if not math.isfinite(p_wait) or p_wait < 0:
        p_wait = 0.0
        
    wait_sec = p_wait / (c * mu_service - lambda_rate)
    if not math.isfinite(wait_sec) or wait_sec < 0:
        wait_sec = 0.0
        
    queue_len = lambda_rate * wait_sec
    return c, float(utilization), float(wait_sec), float(queue_len)


class TelemedicinePipelineSimulator:
    def __init__(
        self,
        annual_patients=100000,
        working_days=300,
        hours_per_day=8,
        images_per_patient=2,
        acquisition_sec_per_image=45.0,
        camera_count=4,
        image_size_mb=2.5,
        bandwidth_mbps=4.0,
        network_latency_sec=0.15,
        good_image_rate=0.82,
        unusable_image_rate=0.18,
        ai_latency_sec=0.457,  # Measured hardware latency from benchmark_throughput.py
        ai_workers=2,
        referable_prevalence=0.18,
        review_sec_per_case=30.0,
        ophthalmologist_count=4,
        target_utilization=0.75
    ):
        self.annual_patients = annual_patients
        self.working_days = working_days
        self.hours_per_day = hours_per_day
        self.total_operating_hours = working_days * hours_per_day
        
        self.images_per_patient = images_per_patient
        self.acquisition_sec_per_image = acquisition_sec_per_image
        self.camera_count = camera_count
        
        self.image_size_mb = image_size_mb
        self.bandwidth_mbps = bandwidth_mbps
        self.network_latency_sec = network_latency_sec
        
        self.good_image_rate = good_image_rate
        self.unusable_image_rate = unusable_image_rate
        
        self.ai_latency_sec = ai_latency_sec
        self.ai_workers = ai_workers
        
        self.referable_prevalence = referable_prevalence
        self.review_sec_per_case = review_sec_per_case
        self.ophthalmologist_count = ophthalmologist_count
        self.target_utilization = target_utilization

    def run_simulation(self):
        # 1. Patient & Image Arrival Rates
        lambda_patients_hr = self.annual_patients / self.total_operating_hours
        lambda_patients_sec = lambda_patients_hr / 3600.0
        lambda_images_sec = lambda_patients_sec * self.images_per_patient
        
        # 2. Camera Acquisition Stage
        mu_camera = 1.0 / self.acquisition_sec_per_image
        c_cam, u_cam, wait_cam_sec, q_cam = solve_mmc_queue(
            lambda_images_sec, mu_camera, self.camera_count
        )
        
        # 3. Rural Network Transmission Stage
        # Upload time: (Size in Megabits / Bandwidth in Mbps) + roundtrip ping
        upload_sec = (self.image_size_mb * 8.0) / self.bandwidth_mbps + self.network_latency_sec
        mu_network = 1.0 / upload_sec
        network_channels = max(1, math.ceil(self.bandwidth_mbps / 2.0))
        c_net, u_net, wait_net_sec, q_net = solve_mmc_queue(
            lambda_images_sec, mu_network, network_channels
        )
        
        # 4. IQA Gating & Recapture Load
        # 18% of images need prompt recapture
        total_images_acquired = self.annual_patients * self.images_per_patient * (1.0 + self.unusable_image_rate)
        total_recaptures = self.annual_patients * self.images_per_patient * self.unusable_image_rate
        effective_image_rate = lambda_images_sec * (1.0 + self.unusable_image_rate)
        
        # 5. Automated AI Processing Stage
        mu_ai = 1.0 / self.ai_latency_sec
        c_ai, u_ai, wait_ai_sec, q_ai = solve_mmc_queue(
            effective_image_rate, mu_ai, self.ai_workers
        )
        
        # 6. Human Review: Ophthalmologist Telemedicine Pool
        # Only referable/flagged cases proceed to human review
        lambda_review_sec = lambda_patients_sec * self.referable_prevalence
        mu_review = 1.0 / self.review_sec_per_case
        c_rev, u_rev, wait_rev_sec, q_rev = solve_mmc_queue(
            lambda_review_sec, mu_review, self.ophthalmologist_count
        )
        
        # End-to-end cumulative metrics
        total_service_time_sec = (
            self.acquisition_sec_per_image * self.images_per_patient
            + upload_sec * self.images_per_patient
            + self.ai_latency_sec * self.images_per_patient
            + self.review_sec_per_case * self.referable_prevalence
        )
        total_wait_sec = (
            wait_cam_sec * self.images_per_patient
            + wait_net_sec * self.images_per_patient
            + wait_ai_sec * self.images_per_patient
            + wait_rev_sec
        )
        
        utils = {"Camera Acquisition": u_cam, "Network Bandwidth": u_net, "AI Processing": u_ai, "Ophthalmologist Review": u_rev}
        bottleneck = max(utils, key=utils.get)
        max_queue = max(q_cam, q_net, q_ai, q_rev)
        
        results = {
            "annual_patients": self.annual_patients,
            "total_images_acquired": int(round(total_images_acquired)),
            "total_recaptures": int(round(total_recaptures)),
            "total_ai_screenings": int(round(total_images_acquired)),
            "total_human_reviews": int(round(self.annual_patients * self.referable_prevalence)),
            "total_referrals": int(round(self.annual_patients * self.referable_prevalence)),
            "arrival_rate": {
                "patients_per_year": self.annual_patients,
                "patients_per_day": round(self.annual_patients / self.working_days, 1),
                "patients_per_hour": round(lambda_patients_hr, 2),
                "patients_per_minute": round(lambda_patients_hr / 60.0, 3),
                "images_per_hour": round(lambda_images_sec * 3600.0, 1)
            },
            "subsystems": {
                "camera_acquisition": {
                    "cameras": self.camera_count,
                    "service_time_sec": self.acquisition_sec_per_image,
                    "utilization": round(u_cam, 4),
                    "waiting_time_sec": round(wait_cam_sec, 2),
                    "queue_length": round(q_cam, 2)
                },
                "network_transmission": {
                    "bandwidth_mbps": self.bandwidth_mbps,
                    "upload_time_sec": round(upload_sec, 2),
                    "utilization": round(u_net, 4),
                    "waiting_time_sec": round(wait_net_sec, 2),
                    "queue_length": round(q_net, 2)
                },
                "ai_processing": {
                    "workers": self.ai_workers,
                    "inference_latency_sec": round(self.ai_latency_sec, 3),
                    "utilization": round(u_ai, 4),
                    "waiting_time_sec": round(wait_ai_sec, 2),
                    "queue_length": round(q_ai, 2)
                },
                "ophthalmologist_review": {
                    "ophthalmologists": self.ophthalmologist_count,
                    "review_time_sec": round(self.review_sec_per_case, 1),
                    "utilization": round(u_rev, 4),
                    "waiting_time_sec": round(wait_rev_sec, 2),
                    "queue_length": round(q_rev, 2)
                }
            },
            "end_to_end": {
                "service_time_seconds": round(total_service_time_sec, 2),
                "average_wait_time_minutes": round(total_wait_sec / 60.0, 2),
                "maximum_queue_length": int(round(max_queue)),
                "primary_bottleneck": bottleneck
            }
        }
        return results


def run_all_scenarios():
    print("=" * 75)
    print("RUNNING 5 TELEMEDICINE OPERATIONAL SCENARIOS (A, B, C, D, E)")
    print("=" * 75)
    
    scenarios = {
        "A": {
            "name": "Scenario A: Small Rural PHC",
            "params": {"annual_patients": 5000, "camera_count": 1, "bandwidth_mbps": 1.0, "ai_workers": 1, "ophthalmologist_count": 1}
        },
        "B": {
            "name": "Scenario B: Medium PHC",
            "params": {"annual_patients": 25000, "camera_count": 2, "bandwidth_mbps": 4.0, "ai_workers": 1, "ophthalmologist_count": 2}
        },
        "C": {
            "name": "Scenario C: District-Level Screening (>=100k pts/yr)",
            "params": {"annual_patients": 100000, "camera_count": 4, "bandwidth_mbps": 8.0, "ai_workers": 2, "ophthalmologist_count": 4}
        },
        "D": {
            "name": "Scenario D: Peak Load Surge (Morning Camp)",
            "params": {"annual_patients": 100000, "hours_per_day": 6, "camera_count": 4, "bandwidth_mbps": 4.0, "ai_workers": 2, "ophthalmologist_count": 4}
        },
        "E": {
            "name": "Scenario E: Improved Infrastructure (Edge AI + 15 Mbps)",
            "params": {"annual_patients": 100000, "camera_count": 5, "bandwidth_mbps": 15.0, "ai_workers": 4, "ophthalmologist_count": 5}
        }
    }
    
    scenario_outputs = {}
    print(f"\n{'Scenario':<28} | {'Volume':<8} | {'Cams':<4} | {'Bw':<6} | {'Docs':<4} | {'Wait(min)':<9} | {'MaxQ':<4} | {'Bottleneck'}")
    print("-" * 95)
    
    for key, sc in scenarios.items():
        sim = TelemedicinePipelineSimulator(**sc["params"])
        out = sim.run_simulation()
        scenario_outputs[key] = out
        
        cams = out["subsystems"]["camera_acquisition"]["cameras"]
        bw = out["subsystems"]["network_transmission"]["bandwidth_mbps"]
        docs = out["subsystems"]["ophthalmologist_review"]["ophthalmologists"]
        wait_m = out["end_to_end"]["average_wait_time_minutes"]
        max_q = out["end_to_end"]["maximum_queue_length"]
        bn = out["end_to_end"]["primary_bottleneck"]
        
        print(f"{sc['name']:<28} | {out['annual_patients']:<8} | {cams:<4} | {bw:<4}Mb | {docs:<4} | {wait_m:<9.2f} | {max_q:<4} | {bn}")
        
    print("-" * 95)
    return scenario_outputs


def run_bandwidth_sweep():
    """Generates Bandwidth vs Upload Time, Throughput, and Queue Length."""
    bandwidths = [0.5, 1.0, 2.0, 4.0, 8.0, 10.0, 15.0, 25.0, 50.0]
    sweep_data = []
    
    for bw in bandwidths:
        sim = TelemedicinePipelineSimulator(annual_patients=100000, bandwidth_mbps=bw, camera_count=4, ophthalmologist_count=4)
        res = sim.run_simulation()
        net = res["subsystems"]["network_transmission"]
        upload_time = net["upload_time_sec"]
        throughput_imgs_hr = 3600.0 / upload_time if upload_time > 0 else 0
        
        sweep_data.append({
            "bandwidth_mbps": bw,
            "upload_time_sec": upload_time,
            "images_per_hour": round(throughput_imgs_hr, 1),
            "utilization": net["utilization"],
            "waiting_time_sec": net["waiting_time_sec"],
            "queue_length": net["queue_length"]
        })
    return sweep_data


def run_resource_optimization(target_volume=100000, max_wait_min=15.0, max_util=0.75):
    print("\n" + "=" * 75)
    print(f"RUNNING RESOURCE OPTIMIZATION FOR >= {target_volume:,} PATIENTS/YEAR")
    print(f"Targets: Average Wait <= {max_wait_min} min | Max Subsystem Utilization <= {max_util*100:.0f}%")
    print("=" * 75)
    
    camera_grid = range(2, 9)
    ai_worker_grid = range(1, 6)
    doctor_grid = range(2, 9)
    bandwidth_grid = [2.0, 4.0, 8.0, 12.0, 16.0]
    
    feasible_configs = []
    best_cost = float("inf")
    best_config = None
    
    for c in camera_grid:
        for w in ai_worker_grid:
            for d in doctor_grid:
                for bw in bandwidth_grid:
                    sim = TelemedicinePipelineSimulator(
                        annual_patients=target_volume,
                        camera_count=c,
                        ai_workers=w,
                        ophthalmologist_count=d,
                        bandwidth_mbps=bw
                    )
                    res = sim.run_simulation()
                    
                    u_cam = res["subsystems"]["camera_acquisition"]["utilization"]
                    u_net = res["subsystems"]["network_transmission"]["utilization"]
                    u_ai = res["subsystems"]["ai_processing"]["utilization"]
                    u_rev = res["subsystems"]["ophthalmologist_review"]["utilization"]
                    wait_m = res["end_to_end"]["average_wait_time_minutes"]
                    
                    if (
                        u_cam <= max_util
                        and u_net <= max_util
                        and u_ai <= max_util
                        and u_rev <= max_util
                        and wait_m <= max_wait_min
                    ):
                        cost = (c * 10) + (d * 40) + (w * 5) + (bw * 2)
                        cfg = {
                            "cameras": c,
                            "ai_workers": w,
                            "doctors": d,
                            "bandwidth_mbps": bw,
                            "cost_index": cost,
                            "wait_time_min": wait_m,
                            "max_queue": res["end_to_end"]["maximum_queue_length"],
                            "util_camera": u_cam,
                            "util_network": u_net,
                            "util_ai": u_ai,
                            "util_review": u_rev
                        }
                        feasible_configs.append(cfg)
                        if cost < best_cost:
                            best_cost = cost
                            best_config = cfg
                            
    if not best_config and feasible_configs:
        best_config = feasible_configs[0]
        
    print(f"Feasible Configurations Found: {len(feasible_configs)}")
    if best_config:
        print(f"Optimal Configuration:")
        print(f"  - Cameras:                    {best_config['cameras']} units (Util: {best_config['util_camera']*100:.1f}%)")
        print(f"  - AI Workers:                 {best_config['ai_workers']} instances (Util: {best_config['util_ai']*100:.1f}%)")
        print(f"  - Ophthalmologists:           {best_config['doctors']} doctors (Util: {best_config['util_review']*100:.1f}%)")
        print(f"  - Bandwidth:                  {best_config['bandwidth_mbps']} Mbps (Util: {best_config['util_network']*100:.1f}%)")
        print(f"  - Average Wait Time:          {best_config['wait_time_min']:.2f} minutes")
        print(f"  - Maximum Queue:              {best_config['max_queue']} patients")
        print(f"  - Annual Screening Capacity:  >= {target_volume:,} patients/year")
        
    return {
        "target_volume": target_volume,
        "feasible_count": len(feasible_configs),
        "optimal_configuration": best_config,
        "top_alternatives": feasible_configs[:4]
    }


def main():
    # 1. Run 100,000+ District Screening Base Simulation
    district_sim = TelemedicinePipelineSimulator(annual_patients=100000, camera_count=4, bandwidth_mbps=8.0, ai_workers=2, ophthalmologist_count=4)
    district_results = district_sim.run_simulation()
    
    # 2. Run Scenarios A through E
    scenario_results = run_all_scenarios()
    
    # 3. Run Bandwidth Sweep
    bw_sweep = run_bandwidth_sweep()
    
    # 4. Run Resource Optimization
    opt_results = run_resource_optimization(target_volume=100000)
    
    # Export files
    out_dir = os.path.join(ROOT_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)
    matlab_dir = os.path.join(ROOT_DIR, "matlab", "results")
    os.makedirs(os.path.join(matlab_dir, "throughput"), exist_ok=True)
    os.makedirs(os.path.join(matlab_dir, "bandwidth"), exist_ok=True)
    os.makedirs(os.path.join(matlab_dir, "queue"), exist_ok=True)
    os.makedirs(os.path.join(matlab_dir, "resource_optimization"), exist_ok=True)
    
    # Save district_screening_summary.json
    with open(os.path.join(out_dir, "district_screening_summary.json"), "w") as f:
        json.dump(district_results, f, indent=2)
    with open(os.path.join(matlab_dir, "district_screening_summary.json"), "w") as f:
        json.dump(district_results, f, indent=2)
        
    # Save district_screening_summary.csv
    csv_rows = [
        ["Metric", "Value", "Unit"],
        ["AnnualPatients", district_results["annual_patients"], "patients/year"],
        ["TotalImagesAcquired", district_results["total_images_acquired"], "images/year"],
        ["TotalRecaptures", district_results["total_recaptures"], "images/year"],
        ["TotalAIScreenings", district_results["total_ai_screenings"], "screenings/year"],
        ["TotalHumanReviews", district_results["total_human_reviews"], "reviews/year"],
        ["TotalReferrals", district_results["total_referrals"], "referrals/year"],
        ["CameraCount", district_results["subsystems"]["camera_acquisition"]["cameras"], "units"],
        ["CameraUtilization", district_results["subsystems"]["camera_acquisition"]["utilization"], "fraction"],
        ["BandwidthMbps", district_results["subsystems"]["network_transmission"]["bandwidth_mbps"], "Mbps"],
        ["NetworkUtilization", district_results["subsystems"]["network_transmission"]["utilization"], "fraction"],
        ["AIWorkers", district_results["subsystems"]["ai_processing"]["workers"], "threads"],
        ["AIUtilization", district_results["subsystems"]["ai_processing"]["utilization"], "fraction"],
        ["Ophthalmologists", district_results["subsystems"]["ophthalmologist_review"]["ophthalmologists"], "doctors"],
        ["ReviewerUtilization", district_results["subsystems"]["ophthalmologist_review"]["utilization"], "fraction"],
        ["AverageWaitTimeMin", district_results["end_to_end"]["average_wait_time_minutes"], "minutes"],
        ["MaximumQueueLength", district_results["end_to_end"]["maximum_queue_length"], "patients"],
        ["PrimaryBottleneck", district_results["end_to_end"]["primary_bottleneck"], "subsystem"]
    ]
    with open(os.path.join(out_dir, "district_screening_summary.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
    with open(os.path.join(matlab_dir, "district_screening_summary.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
        
    # Save bandwidth_sweep.csv
    with open(os.path.join(matlab_dir, "bandwidth", "bandwidth_sweep.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["bandwidth_mbps", "upload_time_sec", "images_per_hour", "utilization", "waiting_time_sec", "queue_length"])
        writer.writeheader()
        writer.writerows(bw_sweep)
        
    # Save optimization_results.json
    with open(os.path.join(matlab_dir, "resource_optimization", "optimization_results.json"), "w") as f:
        json.dump(opt_results, f, indent=2)
        
    print("\n" + "=" * 75)
    print("SIMULATION EXPORTS COMPLETE!")
    print(f"Saved summary to: {os.path.join(out_dir, 'district_screening_summary.csv')}")
    print(f"Saved JSON to:    {os.path.join(out_dir, 'district_screening_summary.json')}")
    print("=" * 75)


if __name__ == "__main__":
    main()
