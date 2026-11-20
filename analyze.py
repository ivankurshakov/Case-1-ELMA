"""Воспроизводимый поиск отклонений в учебном журнале заявок на закупку.

Запуск: python analyze.py
Источник по умолчанию: данные/students.csv. Никаких сетевых запросов или вызовов ИИ.
Рабочий календарь: пн–пт, 09:00–18:00, UTC+03:00, без праздников и обеда.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

LOCAL = timezone(timedelta(hours=3))
CUTOFF = "2026-07-01T00:00:00+03:00"
CHECK = "Проверка заявки"
MANAGER = "Согласование руководителем"
BUDGET = "Согласование бюджета"
LEGAL = "Проверка договора"
EXECUTION = "Исполнение заявки"
REWORK = "Доработка заявки"
AUDIT = "Внеплановая проверка"
EXPECTED_ROLE = {
    CHECK: "specialist", MANAGER: "manager", BUDGET: "finance",
    LEGAL: "legal", EXECUTION: "procurement", REWORK: "requester", AUDIT: "auditor",
}
MIN_REFERENCE_N = 30
IQR_MULTIPLIER = 6
MEDIAN_MULTIPLIER = 5
FAST_SECONDS = 30
FAST_MEDIAN_RATIO = 0.1
RETURN_LIMIT = 3


def working_clock(value: pd.Timestamp) -> float:
    """Накопленные секунды рабочего календаря до данного момента."""
    t = value.tz_convert(LOCAL)
    days = int(np.busday_count("1970-01-01", t.date().isoformat()))
    seconds = t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6
    partial = max(0.0, min(9 * 3600, seconds - 9 * 3600)) if t.weekday() < 5 else 0
    return days * 9 * 3600 + partial


def working_hours(start: pd.Timestamp, end: pd.Timestamp) -> float:
    if end < start:
        raise ValueError(f"Отрицательная длительность: {start} / {end}")
    return (working_clock(end) - working_clock(start)) / 3600


def independent_working_hours(start: pd.Timestamp, end: pd.Timestamp) -> float:
    """Независимая сверка: суммирование пересечений с каждым рабочим днём."""
    start, end = start.tz_convert(LOCAL), end.tz_convert(LOCAL)
    current = start.date()
    total = 0.0
    while current <= end.date():
        if current.weekday() < 5:
            left = pd.Timestamp(datetime(current.year, current.month, current.day, 9, tzinfo=LOCAL))
            right = left + pd.Timedelta(hours=9)
            total += max(0.0, (min(end, right) - max(start, left)).total_seconds())
        current += timedelta(days=1)
    return total / 3600


def fmt(t: pd.Timestamp) -> str:
    return t.tz_convert(LOCAL).strftime("%d.%m.%Y %H:%M:%S") + " +03:00"


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    # BOM нужен для удобного открытия русского текста в Excel.
    frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n", float_format="%.6f")


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=here / "данные" / "students.csv")
    parser.add_argument("--out", type=Path, default=here)
    parser.add_argument("--cutoff", default=CUTOFF)
    args = parser.parse_args()
    source, out = args.input.resolve(), args.out.resolve()
    evidence = out / "проверка"
    evidence.mkdir(parents=True, exist_ok=True)
    cutoff = pd.Timestamp(args.cutoff).tz_convert(LOCAL)

    data = pd.read_csv(source, encoding="utf-8-sig", keep_default_na=False)
    expected_columns = [
        "ProcessInstanceId", "EventId", "EventIndex", "EventType", "ActivityId",
        "ActivityName", "TaskId", "Timestamp", "UserId", "Role", "Status",
        "Result", "RequestType", "Amount", "TemplateVersion",
    ]
    if list(data.columns) != expected_columns:
        raise ValueError("Набор колонок не совпадает с описанием журнала")
    data["EventIndex"] = pd.to_numeric(data.EventIndex, errors="raise").astype(int)
    data["Amount"] = pd.to_numeric(data.Amount, errors="raise")
    data["ts"] = pd.to_datetime(data.Timestamp, utc=True, errors="raise").dt.tz_convert(LOCAL)
    data = data.sort_values(["ProcessInstanceId", "EventIndex"], kind="stable").reset_index(drop=True)

    checks = {
        "duplicate_event_ids": int(data.EventId.duplicated().sum()),
        "duplicate_process_indices": int(data.duplicated(["ProcessInstanceId", "EventIndex"]).sum()),
        "nonpositive_amount_rows": int(data.Amount.le(0).sum()),
        "noncontiguous_process_indices": 0,
        "reversed_timestamps": 0,
        "unstable_process_attributes": 0,
        "bad_event_contract_rows": 0,
        "invalid_task_pairs": 0,
    }
    essential = ["ProcessInstanceId", "EventId", "EventType", "ActivityId", "ActivityName", "Timestamp", "RequestType", "TemplateVersion"]
    checks["missing_essential_cells"] = int(data[essential].eq("").sum().sum())
    checks["ambiguous_activity_names"] = int(data.groupby("ActivityName").ActivityId.nunique().ne(1).sum())
    checks["ambiguous_activity_ids"] = int(data.groupby("ActivityId").ActivityName.nunique().ne(1).sum())
    status_contract = {
        "process_started": ("exec", {""}), "automatic_completed": ("completed", {""}),
        "task_created": ("in_progress", {""}), "task_completed": ("closed", {"accept", "return", "reject"}),
        "process_completed": ("done", {"fulfilled", "rejected"}),
    }
    for r in data.itertuples():
        rule = status_contract.get(r.EventType)
        bad = rule is None or r.Status != rule[0] or r.Result not in rule[1]
        if r.EventType == "task_created":
            bad |= r.TaskId == "" or r.UserId != "" or r.Role != ""
        elif r.EventType == "task_completed":
            bad |= r.TaskId == "" or r.UserId == "" or r.Role == ""
        elif r.EventType == "process_started":
            bad |= r.UserId == "" or r.Role != "requester" or r.TaskId != ""
        elif r.EventType == "automatic_completed":
            bad |= r.Role != "system" or r.UserId != "" or r.TaskId != ""
        elif r.EventType == "process_completed":
            bad |= r.Role != "" or r.UserId != "" or r.TaskId != ""
        checks["bad_event_contract_rows"] += int(bad)
    for _, g in data.groupby("ProcessInstanceId", sort=False):
        checks["noncontiguous_process_indices"] += int(g.EventIndex.tolist() != list(range(1, len(g) + 1)))
        checks["reversed_timestamps"] += int(g.ts.diff().dropna().lt(pd.Timedelta(0)).any())
        checks["unstable_process_attributes"] += int(g[["RequestType", "Amount", "TemplateVersion"]].nunique().gt(1).any())
    for _, g in data.loc[data.TaskId.ne("")].groupby("TaskId"):
        kinds = g.EventType.tolist()
        bad = kinds not in [["task_created", "task_completed"], ["task_created"]]
        bad |= g.ProcessInstanceId.nunique() != 1 or g.ActivityId.nunique() != 1
        checks["invalid_task_pairs"] += int(bad)
    if any(checks.values()):
        raise ValueError("Обнаружены ошибки структуры; требуется разбор до анализа: " + str(checks))

    late = data[data.ts.gt(cutoff)]
    observed = data[data.ts.le(cutoff)].copy()
    created = observed[observed.EventType.eq("task_created")]
    completed = observed[observed.EventType.eq("task_completed")]
    tasks = created.merge(
        completed[["ProcessInstanceId", "TaskId", "ts", "UserId", "Role", "Result", "EventIndex"]],
        on=["ProcessInstanceId", "TaskId"], how="left", validate="one_to_one", suffixes=("_created", "_completed"),
    )
    tasks["IsOpen"] = tasks.ts_completed.isna()
    tasks["ObservedEnd"] = tasks.ts_completed.fillna(cutoff)
    tasks["CalendarHours"] = (tasks.ObservedEnd - tasks.ts_created).dt.total_seconds() / 3600
    tasks["WorkingHours"] = [working_hours(a, b) for a, b in zip(tasks.ts_created, tasks.ObservedEnd)]
    # При IsOpen=True значения являются нижней границей до среза, а не полным сроком.

    stats = []
    baseline = {}
    for key, g in tasks.loc[~tasks.IsOpen].groupby(["RequestType", "ActivityName"]):
        values = g.WorkingHours
        q1, med, q3 = (float(values.quantile(q)) for q in (.25, .5, .75))
        high = max(q3 + IQR_MULTIPLIER * (q3 - q1), MEDIAN_MULTIPLIER * med)
        record = dict(RequestType=key[0], ActivityName=key[1], N=len(g), Q1=q1,
            MedianWorkingHours=med, Q3=q3, P95WorkingHours=float(values.quantile(.95)),
            P99WorkingHours=float(values.quantile(.99)), MedianCalendarHours=float(g.CalendarHours.median()),
            UpperWorkingHours=high, TimingRuleEnabled=len(g) >= MIN_REFERENCE_N)
        baseline[key] = record
        stats.append(record)

    findings = []
    categories = {
        "missing_manager": "Нарушение обязательного маршрута",
        "wrong_role": "Несоответствие роли этапу",
        "execution_before_approval": "Нарушение порядка согласований",
        "slow_task": "Статистический кандидат для проверки",
        "stale_open_task": "Незавершённая заявка с большим ожиданием",
        "rapid_task_chain": "Необычная скорость требует проверки",
        "excessive_returns": "Допустимый маршрут с необычным числом возвратов",
        "extra_review": "Дополнительная проверка требует объяснения",
        "unscheduled_audit": "Внеплановый контроль требует объяснения",
        "events_after_cutoff": "Несоответствие выгрузки заявленному срезу",
    }

    def add(pid: str, kind: str, explanation: str) -> None:
        findings.append(dict(ProcessInstanceId=pid, AnomalyType=kind, Explanation=explanation))

    for pid, g in late.groupby("ProcessInstanceId", sort=True):
        first, last = g.iloc[0], g.iloc[-1]
        add(pid, "events_after_cutoff", f"Качество данных: число событий позже заявленного среза {fmt(cutoff)} — {len(g)}. "
            f"Первое: EventIndex={first.EventIndex}, {first.EventType}, {fmt(first.ts)}; "
            f"последнее: EventIndex={last.EventIndex}, {last.EventType}, {fmt(last.ts)}. "
            "Эти события исключены из состояния на срез; требуется уточнить дату выгрузки. Нарушение процесса из этого не следует.")

    process_rows = []
    normal_sequences = Counter()
    lifecycle_errors = []
    return_distribution = completed[(completed.ActivityName == MANAGER) & (completed.Result == 'return')].groupby('ProcessInstanceId').size().reindex(observed.ProcessInstanceId.unique(), fill_value=0).value_counts()
    audit_process_count = completed.loc[completed.ActivityName.eq(AUDIT),'ProcessInstanceId'].nunique()
    ordinary_min_seconds = float(tasks.loc[(~tasks.IsOpen) & tasks.CalendarHours.gt(FAST_SECONDS/3600),'CalendarHours'].min()*3600)
    for pid, g in observed.groupby("ProcessInstanceId", sort=True):
        start = g.iloc[0]
        closed = g[g.EventType.eq("task_completed")]
        done = g[g.EventType.eq("process_completed")]
        process_tasks = tasks[tasks.ProcessInstanceId.eq(pid)]
        outcome = done.iloc[-1].Result if len(done) else "open"
        end = done.iloc[-1].ts if len(done) else cutoff
        route = " → ".join(closed.ActivityName + " [" + closed.Result + "]")
        normal_sequences[(start.RequestType, outcome, route)] += 1

        if len(done) > 1 or sum(g.EventType.eq("process_started")) != 1 or sum(g.EventType.eq("automatic_completed")) != 1:
            lifecycle_errors.append(pid)
        if len(done) and (g.iloc[-1].EventType != "process_completed" or process_tasks.IsOpen.any()):
            lifecycle_errors.append(pid)
        if outcome == "rejected":
            if len(closed) == 0 or closed.iloc[-1].ActivityName != MANAGER or closed.iloc[-1].Result != "reject":
                lifecycle_errors.append(pid)
        if outcome == "fulfilled":
            mandatory = [CHECK, MANAGER, BUDGET] + ([LEGAL] if start.RequestType == "contract" else []) + [EXECUTION]
            accepted = set(closed.loc[closed.Result.eq("accept"), "ActivityName"])
            missing = [stage for stage in mandatory if stage not in accepted]
            if missing:
                # В предоставленном наборе отсутствует только руководитель; другие пропуски требуют нового разбора.
                if missing != [MANAGER]:
                    raise ValueError(f"Не предусмотренный классификатором пропуск: {pid} {missing}")
                budget_event = closed[closed.ActivityName.eq(BUDGET)].iloc[0]
                exec_event = closed[closed.ActivityName.eq(EXECUTION)].iloc[0]
                add(pid, "missing_manager", f"Нарушение схемы: процесс завершён fulfilled (EventIndex={done.iloc[-1].EventIndex}), "
                    f"но этап «{MANAGER}» не создавался и не закрывался. "
                    f"«{BUDGET}» закрыт accept при EventIndex={budget_event.EventIndex}; "
                    f"«{EXECUTION}» — при EventIndex={exec_event.EventIndex}. Согласование руководителя обязательно для {start.RequestType}.")

        for r in closed.itertuples():
            expected = EXPECTED_ROLE[r.ActivityName]
            if r.Role != expected:
                self_approved = r.UserId == start.UserId
                add(pid, "wrong_role", f"Несоответствие роли: «{r.ActivityName}», TaskId={r.TaskId}, "
                    f"EventIndex={r.EventIndex}, {fmt(r.ts)}, Result={r.Result}. "
                    f"Фактический UserId={r.UserId}, Role={r.Role}; по схеме ожидается {expected}. "
                    + ("UserId совпадает с инициатором: есть также признак самосогласования. " if self_approved else "UserId отличается от инициатора; самосогласование не утверждается. ")
                    + "Нужно проверить полномочия и замещение.")

        for execution in g[(g.EventType == "task_created") & (g.ActivityName == EXECUTION)].itertuples():
            prerequisites = [CHECK, MANAGER, BUDGET] + ([LEGAL] if start.RequestType == "contract" else [])
            late_approvals = []
            for stage in prerequisites:
                approvals = closed[(closed.ActivityName == stage) & (closed.Result == "accept")]
                if not approvals.empty and not approvals.EventIndex.lt(execution.EventIndex).any():
                    approval = approvals.iloc[0]
                    late_approvals.append(f"«{stage}» закрыт лишь EventIndex={approval.EventIndex}, {fmt(approval.ts)}")
            if late_approvals:
                exec_close = closed[closed.TaskId.eq(execution.TaskId)].iloc[0]
                add(pid, "execution_before_approval", f"Нарушение порядка: «{EXECUTION}» создан при EventIndex={execution.EventIndex}, "
                    f"{fmt(execution.ts)}, и закрыт при EventIndex={exec_close.EventIndex}, {fmt(exec_close.ts)}; "
                    + "; ".join(late_approvals) + ". Эти согласования должны предшествовать исполнению.")

        returns = closed[(closed.ActivityName == MANAGER) & (closed.Result == "return")]
        # Возврат должен сопровождаться доработкой и новой проверкой перед следующим решением.
        for returned in returns.itertuples():
            following = closed[closed.EventIndex.gt(returned.EventIndex)]
            if following.ActivityName.tolist()[:3] != [REWORK, CHECK, MANAGER]:
                lifecycle_errors.append(pid)
        if len(returns) >= RETURN_LIMIT:
            add(pid, "excessive_returns", f"Необычный, но разрешённый маршрут: {len(returns)} решений return руководителя, "
                f"{sum(closed.ActivityName.eq(REWORK))} доработок и {sum(closed.ActivityName.eq(CHECK))} проверок заявки. "
                f"Возвраты при EventIndex={','.join(map(str, returns.EventIndex.tolist()))}. "
                f"В выборке у {return_distribution.get(0,0)} процессов нет возврата, у {return_distribution.get(1,0)} — один; критерий кандидата: не менее {RETURN_LIMIT}. "
                "Нужно выяснить причину повторных исправлений; сам возврат нарушением не является.")

        reviews = closed[closed.ActivityName.eq(CHECK)]
        allowed_reviews = 1 + len(returns)
        if len(reviews) > allowed_reviews:
            add(pid, "extra_review", f"Дополнительная проверка: число закрытий «{CHECK}» — {len(reviews)}, возвратов руководителя — {len(returns)}; "
                f"ожидаемое число проверок по маршруту — {allowed_reviews}. EventIndex={','.join(map(str, reviews.EventIndex.tolist()))}; "
                f"TaskId={','.join(reviews.TaskId.tolist())}. Разные TaskId исключают повтор одной записи. "
                "Нужно объяснение дополнительной проверки; один повтор ActivityId не доказывает ошибку.")

        audits = closed[closed.ActivityName.eq(AUDIT)]
        if len(audits):
            add(pid, "unscheduled_audit", f"Кандидат для ручной проверки: {len(audits)} этапа «{AUDIT}», "
                f"EventIndex={','.join(map(str, audits.EventIndex.tolist()))}, роли={','.join(sorted(set(audits.Role)))}. "
                f"Такой этап есть у {audit_process_count} из {observed.ProcessInstanceId.nunique()} заявок. Auditor является допустимой ролью; "
                "редкость не доказывает нарушение. Проверить основание и результаты внепланового контроля.")

        fast = []
        for task in process_tasks.itertuples():
            reference = baseline[(task.RequestType, task.ActivityName)]
            threshold = reference["UpperWorkingHours"]
            if reference["TimingRuleEnabled"] and task.WorkingHours > threshold:
                if task.IsOpen:
                    add(pid, "stale_open_task", f"На срез {fmt(cutoff)} задание «{task.ActivityName}» "
                        f"TaskId={task.TaskId}, созданное {fmt(task.ts_created)} (EventIndex={task.EventIndex_created}), не закрыто. "
                        f"Наблюдаемое ожидание не менее {task.WorkingHours:.2f} рабочего ч / {task.CalendarHours:.2f} календарного ч; "
                        f"медиана закрытых {task.RequestType}: {reference['MedianWorkingHours']:.3f} рабочего ч, "
                        f"порог {threshold:.3f} ч (n={reference['N']}). "
                        "Это нижняя граница длительности; открытость сама по себе не считается отклонением. Проверить приостановку и архивирование.")
                else:
                    add(pid, "slow_task", f"Статистический кандидат: «{task.ActivityName}», TaskId={task.TaskId}, "
                        f"EventIndex={task.EventIndex_created}/{int(task.EventIndex_completed)}, "
                        f"создано {fmt(task.ts_created)}, закрыто {fmt(task.ts_completed)}. "
                        f"Длительность {task.WorkingHours:.2f} рабочего ч / {task.CalendarHours:.2f} календарного ч; "
                        f"для {task.RequestType} медиана {reference['MedianWorkingHours']:.3f} ч, "
                        f"порог max(Q3+6×IQR; 5×медиана)={threshold:.3f} ч (n={reference['N']}). "
                        "Проверить причину ожидания; договорного SLA в журнале нет.")
            if not task.IsOpen and task.ActivityName in {CHECK, MANAGER, BUDGET, LEGAL, EXECUTION}:
                if task.CalendarHours * 3600 <= FAST_SECONDS and task.WorkingHours <= FAST_MEDIAN_RATIO * reference["MedianWorkingHours"]:
                    fast.append(task)
        if len({t.ActivityName for t in fast}) >= 2:
            details = "; ".join(f"«{t.ActivityName}» {t.CalendarHours*3600:.0f} с, TaskId={t.TaskId}" for t in fast)
            add(pid, "rapid_task_chain", f"Необычная скорость: число разных обязательных этапов, закрытых каждый не более чем за {FAST_SECONDS} с, — {len(fast)}; "
                f"каждая длительность также не превышает 10% медианы своей группы. {details}. "
                f"У заданий длительностью более {FAST_SECONDS} с минимальная календарная длительность {ordinary_min_seconds:.0f} с. "
                "Проверить импорт истории, автоматическое закрытие или формальное согласование; скорость не доказывает нарушение.")

        process_rows.append(dict(ProcessInstanceId=pid, RequestType=start.RequestType, Amount=start.Amount,
            OutcomeAtCutoff=outcome, StartedAt=start.Timestamp, ObservedEnd=end.isoformat(),
            CalendarHours=(end-start.ts).total_seconds()/3600, WorkingHours=working_hours(start.ts,end),
            ManagerReturns=len(returns), CompletedTasks=len(closed), OpenTasks=int(process_tasks.IsOpen.sum()),
            ExtraReviews=max(0,len(reviews)-allowed_reviews), AuditTasks=len(audits), Route=route))

    if lifecycle_errors:
        raise ValueError("Непокрытое отклонение жизненного цикла: " + str(sorted(set(lifecycle_errors))))
    anomalies = pd.DataFrame(findings, columns=["ProcessInstanceId", "AnomalyType", "Explanation"])
    anomalies = anomalies.sort_values(["ProcessInstanceId", "AnomalyType"], kind="stable").reset_index(drop=True)
    assert not anomalies.duplicated().any()
    assert set(anomalies.ProcessInstanceId) <= set(data.ProcessInstanceId)
    assert not anomalies.Explanation.eq("").any()
    processes = pd.DataFrame(process_rows)
    flagged_ids = set(anomalies.ProcessInstanceId)
    processes["InAnomalies"] = processes.ProcessInstanceId.isin(flagged_ids)

    # Проверка календаря другим способом на всех заданиях и независимых граничных примерах.
    independent = [independent_working_hours(a,b) for a,b in zip(tasks.ts_created,tasks.ObservedEnd)]
    max_calendar_error = float(np.max(np.abs(np.asarray(independent)-tasks.WorkingHours.to_numpy())))
    assert max_calendar_error < 1e-8
    assert working_hours(pd.Timestamp("2026-06-26T17:00:00+03:00"),pd.Timestamp("2026-06-29T10:00:00+03:00")) == 2
    assert working_hours(pd.Timestamp("2026-06-27T10:00:00+03:00"),pd.Timestamp("2026-06-28T17:00:00+03:00")) == 0
    assert working_hours(pd.Timestamp("2026-06-30T17:00:00+03:00"),cutoff) == 1

    sensitivity = {}
    for k,m in [(4,3),(6,5),(8,7)]:
        selected=[]
        for task in tasks.itertuples():
            b=baseline[(task.RequestType,task.ActivityName)]
            h=max(b["Q3"]+k*(b["Q3"]-b["Q1"]),m*b["MedianWorkingHours"])
            if b["TimingRuleEnabled"] and task.WorkingHours>h:
                selected.append(task.ProcessInstanceId)
        sensitivity[f"IQR_{k}_median_{m}"]=dict(processes=len(set(selected)),ids=sorted(set(selected)))
    assert sensitivity['IQR_4_median_3']['ids'] == sensitivity['IQR_6_median_5']['ids'] == sensitivity['IQR_8_median_7']['ids']

    save_csv(anomalies,out / "anomalies.csv")
    save_csv(pd.DataFrame(stats), evidence / "baselines.csv")
    save_csv(processes,evidence / "process_metrics.csv")
    task_export = tasks[["ProcessInstanceId","RequestType","ActivityName","TaskId","ts_created","ts_completed",
        "EventIndex_created","EventIndex_completed","UserId_completed","Role_completed","Result_completed",
        "IsOpen","ObservedEnd","CalendarHours","WorkingHours"]].copy()
    save_csv(task_export,evidence / "task_metrics.csv")

    # Повторное чтение проверяет формат и отсутствие потерь UUID / кириллицы при записи CSV.
    reread=pd.read_csv(out / "anomalies.csv",encoding="utf-8-sig",keep_default_na=False)
    assert reread.equals(anomalies)
    snapshot_completed = observed[observed.EventType.eq("process_completed")]
    review_counts = processes.ManagerReturns.value_counts().sort_index().to_dict()
    summary = {
        "input_file": source.name, "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "cutoff": cutoff.isoformat(), "raw_events":len(data), "snapshot_events":len(observed),
        "processes":int(data.ProcessInstanceId.nunique()), "request_types":processes.RequestType.value_counts().to_dict(),
        "raw_completed_processes":int(data.EventType.eq("process_completed").sum()),
        "snapshot_completed_processes":len(snapshot_completed), "snapshot_outcomes":processes.OutcomeAtCutoff.value_counts().to_dict(),
        "events_after_cutoff":len(late), "processes_with_events_after_cutoff":int(late.ProcessInstanceId.nunique()),
        "snapshot_tasks":len(tasks), "snapshot_closed_tasks":int((~tasks.IsOpen).sum()), "snapshot_open_tasks":int(tasks.IsOpen.sum()),
        "open_processes_without_current_task":int(((processes.OutcomeAtCutoff=='open')&(processes.OpenTasks==0)).sum()),
        "anomaly_rows":len(anomalies), "candidate_processes":int(anomalies.ProcessInstanceId.nunique()),
        "by_type":anomalies.AnomalyType.value_counts().sort_index().to_dict(),
        "processes_without_candidates":int((~processes.InAnomalies).sum()),
        "processes_with_multiple_types":int(anomalies.groupby('ProcessInstanceId').AnomalyType.nunique().gt(1).sum()),
        "classification":categories, "manager_return_counts":{str(k):int(v) for k,v in review_counts.items()},
        "data_quality_checks":checks,
        "verification":{"all_process_ids_exist":True,"duplicate_output_rows":0,"csv_roundtrip_exact":True,
            "calendar_intervals_independently_checked":len(tasks),"calendar_max_abs_error_hours":max_calendar_error,
            "lifecycle_check_passed":True,"timing_sensitivity_same_candidates":True},
        "timing_sensitivity":sensitivity,
        "top_routes":[dict(RequestType=t,Outcome=o,Count=n,Route=r) for (t,o,r),n in normal_sequences.most_common(15)],
        "completed_process_duration_medians":processes[processes.OutcomeAtCutoff.eq('fulfilled')].groupby('RequestType')[["CalendarHours","WorkingHours"]].median().round(6).to_dict('index'),
        "observed_roles":[dict(ActivityName=a,Role=r,Count=int(n)) for (a,r),n in completed.groupby('ActivityName').Role.value_counts().items()],
    }
    (evidence / "summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:summary[k] for k in ["raw_events","snapshot_events","processes","snapshot_outcomes","anomaly_rows","candidate_processes","by_type"]},ensure_ascii=False,indent=2))
    print("Проверки идентификаторов, CSV, жизненного цикла и рабочего календаря пройдены.")


if __name__ == "__main__":
    main()
