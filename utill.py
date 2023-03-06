from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class Helper:

    def view_errors(self, y_true, y_pred):
        print(f'mean_squared_error: {mean_squared_error(y_true, y_pred)}')
        print(f'mean_absolute_error: {mean_absolute_error(y_true, y_pred)}')
        print(f'r2_score: {r2_score(y_true, y_pred)}')
        return

    def create_model(self, x_train, y_train, x_test, y_test, feats, model):
        model.fit(x_train.loc[:, feats], y_train)
        y_pred = model.predict(x_test.loc[:, feats])
        self.view_errors(y_test, y_pred)
        return y_pred
