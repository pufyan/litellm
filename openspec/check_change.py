#!/usr/bin/env python3
"""Детерминированная проверка согласованности OpenSpec-change.

Проверяет ровно те классы ошибок, которые на change pjsip-account-registration
находились шестью проходами ручного ревью подряд. Каждая проверка соответствует
реальной найденной ошибке, а не гипотетической.

Запуск:  python3 openspec/check_change.py <change-name>
Код возврата: 0 — чисто, 1 — найдены проблемы.

Это ДОПОЛНЕНИЕ к `openspec validate --strict`, а не замена: validate проверяет
формат (заголовки, наличие сценариев), этот скрипт — смысловые связи между
артефактами, которые формату не видны.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def requirement_titles(text: str) -> list[str]:
    return re.findall(r"^### Requirement:\s*(.+?)\s*$", text, re.M)


def scenario_titles(text: str) -> list[str]:
    return re.findall(r"^#### Scenario:\s*(.+?)\s*$", text, re.M)


def task_ids(text: str) -> set[str]:
    return set(re.findall(r"^- \[[ x]\] (\d+\.\d+)\s", text, re.M))


def parse_renamed(text: str) -> dict[str, str]:
    """Возвращает {старый заголовок -> новый} из секции RENAMED.

    archive применяет RENAMED ДО MODIFIED (specs-apply.js), поэтому
    заголовок из FROM в основной спеке к моменту MODIFIED уже
    переименован в TO — проверку MODIFIED-против-основной-спеки надо
    делать с учётом этого переименования.
    """
    parts = re.split(r"^## (ADDED|MODIFIED|REMOVED|RENAMED) Requirements\s*$", text, flags=re.M)
    renamed: dict[str, str] = {}
    for i in range(1, len(parts) - 1, 2):
        if parts[i] != "RENAMED":
            continue
        body = parts[i + 1]
        pending_from: str | None = None
        for line in body.splitlines():
            mf = re.match(r"\s*-?\s*FROM:\s*`?###\s*Requirement:\s*(.+?)`?\s*$", line)
            mt = re.match(r"\s*-?\s*TO:\s*`?###\s*Requirement:\s*(.+?)`?\s*$", line)
            if mf:
                pending_from = mf.group(1).strip()
            elif mt and pending_from is not None:
                renamed[pending_from] = mt.group(1).strip()
                pending_from = None
    return renamed


def split_added_modified(text: str) -> tuple[list[str], list[str]]:
    """Возвращает (заголовки ADDED, заголовки MODIFIED)."""
    parts = re.split(r"^## (ADDED|MODIFIED|REMOVED|RENAMED) Requirements\s*$", text, flags=re.M)
    added: list[str] = []
    modified: list[str] = []
    for i in range(1, len(parts) - 1, 2):
        section, body = parts[i], parts[i + 1]
        if section == "ADDED":
            added += requirement_titles(body)
        elif section == "MODIFIED":
            modified += requirement_titles(body)
    return added, modified


def resolve_main_spec(change_dir: Path, df: Path) -> Path:
    """Находит основную спеку для дельты, включая вложенные capability:
    дельта `specs/dialog-controller/spec.md` резолвится в main
    `specs/voip/dialog-controller/spec.md`, а не в несуществующий
    `specs/dialog-controller/spec.md`. Без этого все проверки, читающие
    main spec (дубли/сироты, освобождение унаследованных ссылок на задачи),
    молча деградировали на вложенных capability."""
    capability = df.relative_to(change_dir / "specs").parent
    direct = ROOT / "specs" / capability / "spec.md"
    if direct.exists():
        return direct
    matches = sorted(
        p
        for p in (ROOT / "specs").rglob("spec.md")
        if p.parent != ROOT / "specs" and p.parent.parts[-len(capability.parts):] == capability.parts
    )
    return matches[0] if matches else direct


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check_change.py <change-name>", file=sys.stderr)
        return 2

    change_dir = ROOT / "changes" / sys.argv[1]
    if not change_dir.is_dir():
        print(f"нет такого change: {change_dir}", file=sys.stderr)
        return 2

    tasks_text = read(change_dir / "tasks.md")
    design_text = read(change_dir / "design.md")
    proposal_text = read(change_dir / "proposal.md")
    all_text = tasks_text + design_text + proposal_text

    delta_files = sorted((change_dir / "specs").rglob("spec.md"))
    problems: list[str] = []

    tasks = task_ids(tasks_text)
    decisions = set(re.findall(r"^### (D\d+):", design_text, re.M))

    # --- 0. Формат номера задачи: строго `N.M ` (цифры-точка-цифры-пробел).
    # Ловит буквенные суффиксы (`2.1.t`, `3.4a`) и лишние сегменты (`2.1.3`) -
    # они молча выпадали из task_ids (регэксп не матчил `\s` после номера),
    # из-за чего задача не считалась существующей, а ссылки на её укороченный
    # `N.M` разрешались в чужой кодовый шаг с тем же номером.
    for line in tasks_text.splitlines():
        m = re.match(r"^- \[[ x]\] (\S+)", line)
        if m and not re.fullmatch(r"\d+\.\d+", m.group(1)):
            problems.append(
                f"[формат номера] задача помечена как «{m.group(1)}» — "
                f"номер задачи это строго N.M (без букв и лишних сегментов)"
            )

    # Имена файлов, уже существующих в репозитории: их план не вводит, поэтому
    # ни регистрации в сборке, ни задачи-создателя для них требовать нельзя.
    existing_files: set[str] = set()
    _apps = ROOT.parent / "apps"
    if _apps.is_dir():
        for _f in _apps.rglob("*"):
            if _f.is_file():
                existing_files.add(_f.name)

    # --- 1. Дельта ДОБАВЛЯЕТ требование, уже существующее в основной спеке.
    # Встречалось ТРИЖДЫ: после архивации получилось бы два требования об одном.
    for df in delta_files:
        capability = df.relative_to(change_dir / "specs").parent
        main_spec = resolve_main_spec(change_dir, df)
        if not main_spec.exists():
            continue
        delta_text = read(df)
        main_titles = set(requirement_titles(read(main_spec)))
        renamed = parse_renamed(delta_text)
        # archive применяет RENAMED до MODIFIED: заголовок из TO появится в
        # основной спеке (переименованием из FROM), заголовок из FROM исчезнет.
        effective_main_titles = {renamed.get(t, t) for t in main_titles}
        added, modified = split_added_modified(delta_text)
        for title in added:
            if title in effective_main_titles:
                problems.append(
                    f"[дубль] {capability}: требование «{title}» стоит под ADDED, "
                    f"но уже есть в основной спеке — должно быть MODIFIED"
                )
        for title in modified:
            if title not in effective_main_titles:
                problems.append(
                    f"[сирота] {capability}: требование «{title}» стоит под MODIFIED, "
                    f"но в основной спеке такого заголовка нет — archive оставит старое нетронутым"
                )
        # RENAMED: FROM должен реально существовать в основной спеке.
        for src in renamed:
            if src not in main_titles:
                problems.append(
                    f"[сирота] {capability}: RENAMED FROM «{src}» — в основной спеке "
                    f"такого заголовка нет, переименовывать нечего"
                )

    # --- 2. Ссылка на задачу, которой не существует (ломается перенумерацией).
    # MODIFIED-дельты обязаны нести полный текст требования, включая
    # унаследованные ссылки на задачи архивных ченджей ("Реализуется
    # задачами 6.1...") - такие ссылки валидны, если стоят дословно и в
    # основной спеке; проверяем только то, чего там нет.
    for path in [tasks_text, design_text, proposal_text] + [read(f) for f in delta_files]:
        for ref in re.findall(r"задач[аеиуюмихй]*\s+(\d+\.\d+)|см\.\s+(\d+\.\d+)", path):
            num = ref[0] or ref[1]
            if num and num not in tasks:
                problems.append(f"[битая ссылка] ссылка на задачу {num}, которой нет")
    for f in delta_files:
        main_spec = resolve_main_spec(change_dir, f)
        main_nums = {
            (m[0] or m[1])
            for m in re.findall(r"задач[аеиуюмихй]*\s+(\d+\.\d+)|см\.\s+(\d+\.\d+)", read(main_spec))
        } - {""}
        for ref in re.findall(r"задач[аеиуюмихй]*\s+(\d+\.\d+)|см\.\s+(\d+\.\d+)", read(f)):
            num = ref[0] or ref[1]
            if num and num not in tasks and num in main_nums:
                problems = [p for p in problems if f"ссылка на задачу {num}" not in p]

    # --- 3. Ссылка на решение design, которого не существует.
    for num in set(re.findall(r"\bD(\d+)\b", all_text)):
        if f"D{num}" not in decisions:
            problems.append(f"[битая ссылка] ссылка на решение D{num}, которого нет в design.md")

    # --- 4. Нарушение порядка: задача ТРЕБУЕТ готовым то, что создаётся позже.
    # Даёт ложную зелёную фазу TDD либо ошибку компиляции.
    #
    # Раньше здесь стоял растущий whitelist слов-указателей («см.», «вынесен»,
    # «биндят», «обе »...). Он давал 7 ложных срабатываний на исправном плане,
    # а каждое ложное срабатывание обесценивает инструмент: его перестают
    # читать. Правило перевёрнуто — теперь ищем не «похоже на указатель»,
    # а положительный признак ЗАВИСИМОСТИ: глагол, требующий готовности.
    # Только глаголы, означающие «мне нужно, чтобы ЭТО уже было сделано».
    # «используется в задаче N» сюда НЕ входит: это указатель на потребителя,
    # а не требование готовности (так формулируется фиксация имени файла).
    DEPENDS = re.compile(
        r"(?:опира\w+ на|поверх результата|на основе задачи|"
        r"созданн\w+ задачей|реализованн\w+ задачей|после того как задача)",
    )
    for line in tasks_text.splitlines():
        m = re.match(r"^- \[[ x]\] (\d+)\.(\d+)\s+(.*)", line)
        if not m:
            continue
        grp, idx, body = int(m.group(1)), int(m.group(2)), m.group(3)
        for ref_g, ref_i in re.findall(r"задач[аеиуюмихй]*\s+(\d+)\.(\d+)", body):
            rg, ri = int(ref_g), int(ref_i)
            if (rg, ri) <= (grp, idx):
                continue
            pos = body.find(f"{rg}.{ri}")
            window = body[max(0, pos - 90) : pos + 20]
            if DEPENDS.search(window):
                problems.append(
                    f"[порядок] задача {grp}.{idx} ссылается на {rg}.{ri}, "
                    f"которая идёт позже, и формулировка выглядит как ЗАВИСИМОСТЬ "
                    f"(«{window.strip()[:70]}…») — проверить, не требуется ли несуществующее"
                )

    # --- 4b. Номера задач: дубли и немонотонность внутри группы.
    # Ловится тривиально, а руками пропускается: перенумерация группы 7
    # оставила два «7.7» и два «7.8», и ни одна проверка этого не увидела.
    seen: dict[str, int] = {}
    last_in_group: dict[int, int] = {}
    for line in tasks_text.splitlines():
        m = re.match(r"^- \[[ x]\] (\d+)\.(\d+)\s", line)
        if not m:
            continue
        num = f"{m.group(1)}.{m.group(2)}"
        grp, idx = int(m.group(1)), int(m.group(2))
        seen[num] = seen.get(num, 0) + 1
        if seen[num] == 2:
            problems.append(f"[нумерация] задача {num} объявлена дважды")
        prev = last_in_group.get(grp)
        if prev is not None and idx <= prev:
            problems.append(
                f"[нумерация] в группе {grp} задача {num} идёт после {grp}.{prev} — "
                f"номера внутри группы должны возрастать, иначе исполнение «по номеру» "
                f"нарушит порядок зависимостей"
            )
        last_in_group[grp] = idx

    # --- 5. ТРЕБОВАНИЕ дельты, ни один сценарий которого нигде не привязан к
    # задаче. Было заглушено `pass` и потому не работало вовсе.
    #
    # Проверяем на уровне требования, а не сценария: сноска у каждого сценария
    # не нужна (большинство покрыто очевидной задачей своей группы), но если во
    # всём требовании нет ни ссылки на задачу, ни сноски — оно не проверяется
    # ничем.
    for df in delta_files:
        capability = df.relative_to(change_dir / "specs").parent
        text = read(df)
        if "## REMOVED Requirements" in text and "## ADDED" not in text:
            continue  # удаление требования задачами не покрывается
        blocks = re.split(r"^### Requirement:\s*", text, flags=re.M)[1:]
        for block in blocks:
            title = block.splitlines()[0].strip()
            if re.search(r"задач[аеиуюмихй]*\s+\d+\.\d+", block):
                continue  # где-то внутри требования задача названа
            if re.search(r"^>", block, re.M):
                continue  # сноска объясняет отсрочку/ручную проверку
            problems.append(
                f"[покрытие] {capability}: требование «{title}» не называет ни одной "
                f"задачи и не имеет сноски — непонятно, чем оно проверяется"
            )

    # --- 7. Символ, который план использует, но нигде не создаёт.
    # Так пропал член account_manager_: его звали четыре задачи, а завести
    # не поручила ни одна.
    # Задача считается создателем символа, если её текст содержит глагол
    # заведения. Ищем по ВСЕЙ строке, а не только до первого вхождения имени:
    # «Завести член `X` ... — имя зафиксировано как `X`» упоминает символ
    # дважды, и привязка к первому вхождению теряла настоящего создателя.
    # Обороты про разъезд имён («стоило бы круга правок, как было с `X`»)
    # отсекаются отдельно — там глагол относится к чужой прошлой ошибке.
    CREATES = re.compile(
        r"завести|создать|создаёт|реализовать|добавить|объявить|ввести|перенести", re.I
    )
    MENTIONS_ONLY = re.compile(r"как было с|как чуть не разъехал|разъехалось бы", re.I)

    # Символы, уже существующие в исходниках репозитория: их план не вводит,
    # поэтому задачи-создателя для них требовать нельзя (аналогично existing_files
    # для имён файлов). Без этого символ вроде call_session_ — давний член класса,
    # который change только ИСПОЛЬЗУЕТ, — ложно помечается как висячий.
    existing_symbols: set[str] = set()
    for _f in (list(_apps.rglob("*.hpp")) + list(_apps.rglob("*.cpp"))) if _apps.is_dir() else []:
        try:
            existing_symbols.update(re.findall(r"\b(\w+_)\b", _f.read_text(encoding="utf-8", errors="ignore")))
        except OSError:
            pass

    for symbol in sorted(set(re.findall(r"`(\w+_)`", tasks_text))):
        if symbol in existing_symbols:
            continue  # уже есть в коде — change его использует, а не вводит
        users, creators = 0, 0
        for line in tasks_text.splitlines():
            if not re.match(r"^- \[[ x]\] \d+\.\d+\s", line):
                continue
            # Символ может быть записан внутри более длинного бэктик-фрагмента:
            # `int primary_slot_call_id_ = PJSUA_INVALID_ID`. Ищем по слову.
            if not re.search(rf"`[^`]*\b{re.escape(symbol)}\b[^`]*`", line):
                continue
            users += 1
            if CREATES.search(line) and not MENTIONS_ONLY.search(line):
                creators += 1
        if users >= 2 and creators == 0:
            problems.append(
                f"[висячий символ] «{symbol}» упоминается в {users} задачах, "
                f"но ни одна его не заводит — при реализации окажется, что его негде взять"
            )

    # --- 8. Новый файл заведён, но не зарегистрирован в сборке.
    for m in re.finditer(r"`(?:apps/ai_voip/)?((?:src|tests)/[\w/]+\.(?:cpp|sh))`", tasks_text):
        path = m.group(1)
        base = path.split("/")[-1]
        if base in existing_files:
            continue  # файл уже есть в репозитории — регистрировать не нужно
        stem = base.rsplit(".", 1)[0]
        register_re = re.compile(
            r"(?:зарегистрировать|добавить|включить)[^\n]*CMakeLists[^\n]*"
            + re.escape(stem)
            + r"|(?:зарегистрировать|добавить|включить)[^\n]*"
            + re.escape(stem)
            + r"[^\n]*CMakeLists",
            re.I,
        )
        if not register_re.search(tasks_text):
            problems.append(
                f"[сборка] план заводит новый файл «{path}», но ни одна задача "
                f"не регистрирует «{stem}» в CMakeLists.txt — он не соберётся"
            )

    # --- 9. Конструкции, запрещённые принятыми спеками.
    # boost::asio::post через границу потоков передаёт исполняемый код, что
    # voip/thread-topology и voip/channel-registry запрещают дословно. Ошибка
    # была допущена прямо в design.md и прожила два прохода ревью.
    for fname, text in (("design.md", design_text), ("tasks.md", tasks_text)):
        for line_no, line in enumerate(text.splitlines(), 1):
            if "asio::post" not in line:
                continue
            # Упоминание в объяснении, почему так делать НЕЛЬЗЯ, — не предписание.
            # Отрицание ищем ТОЛЬКО рядом с самим вхождением (окно ±60 симв.):
            # слово-исключение в другом конце длинной строки заглушило бы
            # настоящее предписание, стоящее в её начале.
            pos = line.find("asio::post")
            near = line[max(0, pos - 60) : pos + 60]
            if re.search(
                r"НЕ использовать|запрещ|нарушал|нельзя|передал бы|отвергн|"
                r"не пользуется|не встречается|а не |Почему канал|ни `|нет ни|Отсутствие",
                near,
            ):
                continue
            problems.append(
                f"[запрет] {fname}:{line_no} предписывает boost::asio::post через границу "
                f"потоков — он передаёт исполняемый код, что запрещают "
                f"voip/thread-topology и voip/channel-registry; межпоточный обмен идёт каналом"
            )

    # --- 6. Идентификатор, введённый планом, назван лишь в одном артефакте.
    # Так разъезжались account_manager_, release_account(), PjsipQuillLogWriter.
    # Учитываются только имена, которых в репозитории ЕЩЁ НЕТ: существующий код
    # и чужие API (ThreadRegistry::join_all, application.cpp) план не вводит,
    # и требовать их упоминания в задачах бессмысленно.
    existing = existing_files  # вычислен выше, до проверки 8

    # Классы, уже существующие в исходниках: их план не вводит, а лишь
    # цитирует (например, AccountManager как прецедент инжектирования
    # интервалов). Без этого устоявшийся класс ложно помечается вводимым.
    existing_classes: set[str] = set()
    for _f in (list(_apps.rglob("*.hpp")) + list(_apps.rglob("*.cpp"))) if _apps.is_dir() else []:
        try:
            existing_classes.update(
                re.findall(r"(?:class|struct)\s+([A-Z]\w*)", _f.read_text(encoding="utf-8", errors="ignore"))
            )
        except OSError:
            pass

    introduced = set()
    for pat in (r"`([\w/]*?(\w+\.[ch]pp))`", r"`(\w+::\w+\(\))`", r"`([A-Z]\w+Manager)`"):
        for m in re.findall(pat, all_text):
            introduced.add(m[0] if isinstance(m, tuple) else m)

    # Файлы, которые план лишь ЦИТИРУЕТ как чужие: дореврайтовый код из
    # git-истории и исходники сторонних библиотек. Планом они не вводятся.
    # application.cpp — дореврайтовый код, живёт только в git-истории;
    # endpoint.cpp — исходник pjsua2. Оба лишь цитируются как образец.
    # account.cpp/account_manager.cpp сюда НЕ добавлять: их план вводит.
    foreign = {"application.cpp", "endpoint.cpp"}

    for name in sorted(introduced):
        base = name.split("/")[-1]
        if base in existing or base in foreign:
            continue  # существующий или чужой файл — не вводится этим планом
        if re.fullmatch(r"[A-Z]\w*Manager", name) and name in existing_classes:
            continue  # существующий класс-менеджер — цитируется, а не вводится
        if "::" in name and base.split("::")[0] not in {"AccountManager", "Account"}:
            continue  # метод чужого класса
        places = sum(1 for t in (tasks_text, design_text, proposal_text) if name in t)
        if places == 1 and name in design_text:
            problems.append(
                f"[имя] «{name}» вводится планом, но названо только в design.md — "
                f"задачи его не называют, при реализации имя разъедется"
            )

    if problems:
        print(f"НАЙДЕНО ПРОБЛЕМ: {len(problems)}\n")
        for p in problems:
            print(" ", p)
        return 1

    print("чисто: дублей требований, битых ссылок, нарушений порядка и разъезда имён не найдено")
    return 0


if __name__ == "__main__":
    sys.exit(main())
