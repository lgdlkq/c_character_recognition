train, valid, train_label, valid_label = train_test_split(
            imgs,
            labels,
            shuffle=True,
            test_size=0.1,
            random_state=random.randint(0, 10))
        test, valid, test_label, valid_label = train_test_split(
            valid,
            valid_label,