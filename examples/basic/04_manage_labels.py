from redis_robot_comm import RedisLabelManager


def main():
    label_mgr = RedisLabelManager()

    # Publish available labels
    labels = ["cube", "sphere", "cylinder"]
    print(f"Publishing labels: {labels}")
    label_mgr.publish_labels(labels)

    # Retrieve current labels
    current_labels = label_mgr.get_latest_labels()
    print(f"Current detectable objects: {current_labels}")

    # Add new label dynamically
    print("Adding label 'prism'...")
    label_mgr.add_label("prism")

    updated_labels = label_mgr.get_latest_labels()
    print(f"Updated detectable objects: {updated_labels}")


if __name__ == "__main__":
    main()
