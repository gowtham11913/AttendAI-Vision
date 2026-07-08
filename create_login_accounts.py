from backend.auth_database import (
    initialize_auth_database,
    create_admin_account,
    create_student_account,
)


def main():

    initialize_auth_database()

    print()
    print("AttendAI Vision Account Setup")
    print("=" * 35)

    print()
    print("Create Administrator Account")
    print("-" * 35)

    admin_username = input(
        "Administrator username: "
    ).strip()

    admin_password = input(
        "Administrator password: "
    )

    create_admin_account(
        admin_username,
        admin_password,
    )

    print(
        "Administrator account created successfully."
    )


    print()
    print("Create Student Account")
    print("-" * 35)

    roll_no = input(
        "Registered student roll number: "
    ).strip()

    student_password = input(
        "Student password: "
    )

    create_student_account(
        roll_no,
        student_password,
    )

    print(
        "Student account created successfully."
    )


if __name__ == "__main__":
    main()