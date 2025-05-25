import subprocess
import json

def run_ldapsearch(base_dn):
    """Запускает команду ldapsearch и возвращает ее вывод."""
    cmd = ["ldapsearch", "-Y", "gssapi", "-b", base_dn, "(objectclass=*)"]
    try:
        print(f"Выполнение команды: {' '.join(cmd)}")
        # Убедитесь, что используете корректную кодировку 'utf-8'
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
        stdout, stderr = process.communicate()
        if process.returncode != 0:  # Используйте 0 для успешного завершения процесса
            print(f"Ошибка выполнения ldapsearch: {stderr}")
            return ""
        return stdout
    except Exception as e:
        print(f"Ошибка выполнения ldapsearch: {e}")
        return ""

def parse_ldap_output(ldap_output):
    """Парсит вывод ldapsearch."""
    entries = ldap_output.strip().split("\n\n")
    parsed_data = []
    for entry in entries:
        lines = entry.split("\n")
        dn = lines[0].replace("dn: ", "")
        attributes = {}
        for line in lines[1:]:
            if ": " in line:
                key, value = line.split(": ", 1)  # Разделите только по первому ": "
                if key in attributes:
                    attributes[key].append(value)
                else:
                    attributes[key] = [value]
        parsed_data.append({"dn": dn, "attributes": attributes})
    return parsed_data

def save_to_json(data, filename):
    """Сохраняет данные в JSON файл."""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=1, ensure_ascii=False)

def main():
    ldap_bases = {
        "users": "cn=users,cn=accounts,dc=dmosk,dc=local",
        "groups": "cn=groups,cn=accounts,dc=dmosk,dc=local",
        "roles": "cn=roles,cn=accounts,dc=dmosk,dc=local",
        "privileges": "cn=privileges,cn=pbac,dc=dmosk,dc=local",
        "permissions": "cn=permissions,cn=pbac,dc=dmosk,dc=local"
    }

    for entity, base_dn in ldap_bases.items():
        print(f"Обработка {entity}...")
        ldap_output = run_ldapsearch(base_dn)
        if ldap_output:
            parsed_data = parse_ldap_output(ldap_output)
            save_to_json(parsed_data, f"{entity}.json")
        else:
            print(f"Не удалось получить данные для {entity}.")

if __name__ == "__main__":
    main()
