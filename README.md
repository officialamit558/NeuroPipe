# NeuroLoom Scripts

They breakdown as follows: 
* `create_dataloaders` - a method to prepare and download data if needed.
* `train` - a method containing training model.
* `pred_and_plot_image` - a method use for prediction and plot the performance metrics of the trained model.
* `save_model` - a method help to save the trained model using the pytorch based code base.

# How to use this NeoroLoom

`from NeuroPipe.NeuroLoom import (
    create_dataloaders,
    train,
    pred_and_plot_image,
    save_model,
    walk_through_dir,
    plot_loss_curves,
    plot_decision_boundary,
    plot_predictions,
    accuracy_fn,
    plot_loss_curves,
    pred_and_plot_image,
    set_seeds,
    download_data
)`
