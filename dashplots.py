import math
import numpy as np
import torch
# Plot 
import plotly
import plotly.express as px
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff
import plotly.io as pio
pio.renderers
# dash
import dash
from dash import Dash, dcc, html


#
def traintest(train_losses, train_accs, dev_losses, dev_accs, epochs, steps_per_epoch,figname="fig1", live=False):
  
  # transfer to cpu  and switchback to numpy

  torch.Tensor.ndim = property(lambda self: len(self.shape))  # Fix it
    
  total_iterations = steps_per_epoch
  all_iterations = epochs*steps_per_epoch
  x1 = np.arange(all_iterations)
  x2 = np.arange(total_iterations-1,all_iterations,total_iterations)
  t = np.arange(epochs)


  pio.renderers.default = "pdf+svg+plotly_mimetype+browser+png+vscode+colab+json"
  # pio.kaleido.scope.default_scale=1.0
  # pio.kaleido.scope.default_width=
  # pio.kaleido.scope.default_height=
  # pio.kaleido.scope.mathjax = "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js"
  app = dash.Dash()
  pyfig = go.Figure()
  
  
  # Plot training and validation curves
  # plt.rc('lines', linewidth=1.5, markersize=2)
  # fig, ax1 = plt.subplots(figsize=(6, 4)) # 16 x 9 initial
  # color = 'tab:red'
  # ax1.plot(x1, losses.cpu(), c=color,
  #          alpha=0.25, label="Train Loss")
  # ax1.plot(x2, val_losses, c="red", label="Val. Loss")
  # ax1.set_xlabel("Iterations")
  # ax1.set_ylabel("Avg. Cross-Entropy Loss", c=color)
  # ax1.tick_params(axis='y', labelcolor=color)
  # # ax1.set_ylim(-0.01,3)

  tx1 = go.Scatter(
      x=t, 
      y=train_losses,
      line=dict(color='red', width=1.5), 
      opacity=0.25,
      name='Train Loss',
      yaxis="y"
  )
  tx2 = go.Scatter(
      x=t,
      y=dev_losses, 
      line=dict(color='red', width=1.5), 
      name='Val. Loss',
      yaxis="y"
  )
  pyfig.add_traces([tx1, tx2])

  # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
  # color = 'tab:blue'
  # ax2.plot(x1, accs, c=color, label="Train Acc.", alpha=0.25)
  # ax2.plot(x2, val_accs, c="blue", label="Val. Acc.")
  # ax2.set_ylabel(" Accuracy", c=color)
  # ax2.tick_params(axis='y', labelcolor=color)
  # ax2.set_ylim(-0.01, 1.01)

  # fig.tight_layout()  # otherwise the right y-label is slightly clipped
  # ax1.legend(loc="center")
  # ax2.legend(loc="center right")    
  # plt.savefig(f"../figures/trainplt.pdf", bbox_inches='tight')
  # # plt.show()  

  tx3 = go.Scatter(
      x=t, 
      y=train_accs,
      line=dict(color='blue', width=1.5), 
      opacity = 0.25,
      name='Train Acc.',
      yaxis="y2"
  )
  tx4 = go.Scatter(
      x=t,
      y=dev_accs, 
      line=dict(color='blue', width=1.5), 
      name='Val. Acc.',
      yaxis="y2"
  )
  pyfig.add_traces([tx3, tx4])

  pyfig.update_layout(
      xaxis=dict(
          title="Epochs",
          titlefont=dict(family='Open Sans'),      
      ), 
      yaxis=dict(
          title="Avg. Cross-Entropy Loss",
          titlefont=dict(
              color="red",
              family="Open Sans"
          ),
          tickfont=dict(
              color="red"
      )),
      yaxis2=dict(
          title="Accuracy",
          titlefont=dict(
              color="blue",
              family="Open Sans"
          ),
          tickfont=dict(
              color="blue"
          ),
          anchor="x",
          overlaying="y",
          side="right",
          position=1,
          range=[-0.01,1.01],
      ),
      width=600,
      height=450,
      autosize=True,
      margin=dict(l=5,r=5,b=5,t=5,pad=0
      ),
      legend=dict(
          orientation='h',
          yanchor="bottom",
          xanchor="right",
          x=1.0, y=1.02,
      ),
      # template="plotly_white", #simple_white, none
      plot_bgcolor ='white',
      font=dict(family='Open Sans',size=10,color='black'),
  )
  pyfig.add_shape(
      # Rectangle with reference to the plot
      type="rect",
      xref="paper",
      yref="paper",
      x0=0,y0=0,x1=1.0,y1=1.0,
      line=dict(
          color="black",
          width=1,
      )
  )
  pyfig.update_yaxes(
      showline=True,
      linecolor='black',
      linewidth=0,
      ticks='outside',
      tickwidth=1,
      tickcolor='black',
      # range=[-0.01,1.01],
  )
  pyfig.update_xaxes(
      showline=True,
      linecolor='black',
      linewidth=0,
      ticks='outside',
      tickwidth=1,
      tickcolor='black',
  )
  # pyfig.write_image("fig1.pdf")
  # 300dpi, width = 2inches, height = 1.5inches
  pyfig.write_image(figname+".pdf", width=2*300, height=1.5*300, engine="kaleido")
  pyfig.write_image(figname+".png",engine="kaleido")


  # - offline plotly with mpl
  # pfig = tls.mpl_to_plotly(fig) 
  # plotly.offline.plot(pfig, 'plotly clone')

  app.layout= html.Div([
      dcc.Graph(id='mpl', figure=pyfig, mathjax=True),
  ])

  if live:
    app.run_server(debug=False, port=8922, host='localhost', use_reloader=False)
    
#
def pdiffplot(names,datas,runs,figname="fig1", live=False):
  
  # transfer to cpu  and switchback to numpy

  torch.Tensor.ndim = property(lambda self: len(self.shape))  # Fix it


  pio.renderers.default = "pdf+svg+plotly_mimetype+browser+png+vscode+colab+json"
  # pio.kaleido.scope.default_scale=1.0
  # pio.kaleido.scope.default_width=
  # pio.kaleido.scope.default_height=
  # pio.kaleido.scope.mathjax = "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js"
  app = dash.Dash()
  pyfig = go.Figure()
  
  
  for namei, dfi in zip(names,datas):
        pyfig.add_trace(go.Box(y=dfi,
                name=namei,
                # marker_color=,
                boxmean = True,
                boxpoints = "all", # outliers, suspectedoutliers, all, False
                marker_size=3,
                line_width=1,
                jitter = 0,
                whiskerwidth=0.25,
                # fillcolor = cls,
                width=0.25,
                notched=False,
                notchwidth=0,
               )
        )


  pyfig.update_layout(
      xaxis=dict(
          title="Optimizers",
          titlefont=dict(family='Open Sans'),      
      ), 
      yaxis=dict(
        #   autorange=True,
          title="Test Prediction-Difference",
          titlefont=dict(
            #   color="red",
              family="Open Sans"
          ),
          tickfont=dict(
            #   color="red"
      )),
      width=4*100,
      height=3.5*100,
      autosize=True,
      margin=dict(l=5,r=5,b=5,t=5,pad=0
      ),
      legend=dict(
          orientation='h',
          yanchor="bottom",
          xanchor="right",
          x=1.0, y=1.02,
      ),
      # template="plotly_white", #simple_white, none
      plot_bgcolor ='white',
      font=dict(family='Open Sans',size=8,color='black'),
  )
  pyfig.add_shape(
      # Rectangle with reference to the plot
      type="rect",
      xref="paper",
      yref="paper",
      x0=0,y0=0,x1=1.0,y1=1.0,
      line=dict(
          color="black",
          width=1,
      )
  )
  pyfig.update_yaxes(
      showline=True,
      linecolor='black',
      linewidth=0,
      ticks='outside',
      tickwidth=1,
      tickcolor='black',
      # range=[-0.01,1.01],
  )
  pyfig.update_xaxes(
      showline=True,
      linecolor='black',
      linewidth=0,
      ticks='outside',
      tickwidth=1,
      tickcolor='black',
  )
  # pyfig.write_image("fig1.pdf")
  # 300dpi, width = 2inches, height = 1.5inches
#   , width=1.5*300, height=1.5*300, 
  pyfig.write_image(figname+".pdf",engine="kaleido")
  pyfig.write_image(figname+".png",engine="kaleido")

  # - offline plotly with mpl
  # pfig = tls.mpl_to_plotly(fig) 
  # plotly.offline.plot(pfig, 'plotly clone')

  app.layout= html.Div([
      dcc.Graph(id='mpl', figure=pyfig, mathjax=True),
  ])

  if live:
    app.run_server(debug=False, port=8922, host='localhost', use_reloader=False)

def actpdiffplot(names,datas,runs,figname="fig1", live=False):
  
  # transfer to cpu  and switchback to numpy

  torch.Tensor.ndim = property(lambda self: len(self.shape))  # Fix it


  pio.renderers.default = "pdf+svg+plotly_mimetype+browser+png+vscode+colab+json"
  # pio.kaleido.scope.default_scale=1.0
  # pio.kaleido.scope.default_width=
  # pio.kaleido.scope.default_height=
  # pio.kaleido.scope.mathjax = "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js"
  app = dash.Dash()
  pyfig = go.Figure()
  
  
  for namei, dfi in zip(names,datas):
        pyfig.add_trace(go.Box(y=dfi,
                name=namei,
                # marker_color=,
                boxmean = True,
                boxpoints = "all", # outliers, suspectedoutliers, all, False
                marker_size=3,
                line_width=1,
                jitter = 0,
                whiskerwidth=0.25,
                # fillcolor = cls,
                width=0.25,
                notched=False,
                notchwidth=0,
               )
        )


  pyfig.update_layout(
      xaxis=dict(
          title="Optimizers",
          titlefont=dict(family='Open Sans'),      
      ), 
      yaxis=dict(
        #   autorange=True,
          title="Test Prediction-Difference (Counts)",
          titlefont=dict(
            #   color="red",
              family="Open Sans"
          ),
          tickfont=dict(
            #   color="red"
      )),
      width=4*100,
      height=3.5*100,
      autosize=True,
      margin=dict(l=5,r=5,b=5,t=5,pad=0
      ),
      legend=dict(
          orientation='h',
          yanchor="bottom",
          xanchor="right",
          x=1.0, y=1.02,
      ),
      # template="plotly_white", #simple_white, none
      plot_bgcolor ='white',
      font=dict(family='Open Sans',size=8,color='black'),
  )
  pyfig.add_shape(
      # Rectangle with reference to the plot
      type="rect",
      xref="paper",
      yref="paper",
      x0=0,y0=0,x1=1.0,y1=1.0,
      line=dict(
          color="black",
          width=1,
      )
  )
  pyfig.update_yaxes(
      showline=True,
      linecolor='black',
      linewidth=0,
      ticks='outside',
      tickwidth=1,
      tickcolor='black',
      # range=[-0.01,1.01],
  )
  pyfig.update_xaxes(
      showline=True,
      linecolor='black',
      linewidth=0,
      ticks='outside',
      tickwidth=1,
      tickcolor='black',
  )
  # pyfig.write_image("fig1.pdf")
  # 300dpi, width = 2inches, height = 1.5inches
#   , width=1.5*300, height=1.5*300, 
  pyfig.write_image(figname+".pdf",engine="kaleido")
  pyfig.write_image(figname+".png",engine="kaleido")

  # - offline plotly with mpl
  # pfig = tls.mpl_to_plotly(fig) 
  # plotly.offline.plot(pfig, 'plotly clone')

  app.layout= html.Div([
      dcc.Graph(id='mpl', figure=pyfig, mathjax=True),
  ])

  if live:
    app.run_server(debug=False, port=8922, host='localhost', use_reloader=False)
   
def pvalplot(names,datas,runs,figname="fig1", live=False):
  
  # transfer to cpu  and switchback to numpy

  torch.Tensor.ndim = property(lambda self: len(self.shape))  # Fix it


  pio.renderers.default = "pdf+svg+plotly_mimetype+browser+png+vscode+colab+json"
  # pio.kaleido.scope.default_scale=1.0
  # pio.kaleido.scope.default_width=
  # pio.kaleido.scope.default_height=
  # pio.kaleido.scope.mathjax = "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js"
  app = dash.Dash()
  pyfig = go.Figure()
  
  
  for namei, dfi in zip(names,datas):
        pyfig.add_trace(go.Box(y=dfi,
                name=namei,
                # marker_color=,
                boxmean = True,
                boxpoints = "all", # outliers, suspectedoutliers, all, False
                marker_size=3,
                line_width=1,
                jitter = 0,
                whiskerwidth=0.25,
                # fillcolor = cls,
                width=0.25,
                notched=False,
                notchwidth=0,
               )
        )


  pyfig.update_layout(
      xaxis=dict(
          title="Optimizers",
          titlefont=dict(family='Open Sans'),      
      ), 
      yaxis=dict(
        #   autorange=True,
          title=f"p-values (Test Predictions across {runs} runs)",
          titlefont=dict(
            #   color="red",
              family="Open Sans"
          ),
          tickfont=dict(
            #   color="red"
      )),
      width=4*100,
      height=3.5*100,
      autosize=True,
      margin=dict(l=5,r=5,b=5,t=5,pad=0
      ),
      legend=dict(
          orientation='h',
          yanchor="bottom",
          xanchor="right",
          x=1.0, y=1.02,
      ),
      # template="plotly_white", #simple_white, none
      plot_bgcolor ='white',
      font=dict(family='Open Sans',size=8,color='black'),
  )
  pyfig.add_shape(
      # Rectangle with reference to the plot
      type="rect",
      xref="paper",
      yref="paper",
      x0=0,y0=0,x1=1.0,y1=1.0,
      line=dict(
          color="black",
          width=1,
      )
  )
  pyfig.update_yaxes(
      showline=True,
      linecolor='black',
      linewidth=0,
      ticks='outside',
      tickwidth=1,
      tickcolor='black',
      # range=[-0.01,1.01],
  )
  pyfig.update_xaxes(
      showline=True,
      linecolor='black',
      linewidth=0,
      ticks='outside',
      tickwidth=1,
      tickcolor='black',
  )
  # pyfig.write_image("fig1.pdf")
  # 300dpi, width = 2inches, height = 1.5inches
#   , width=1.5*300, height=1.5*300, 
  pyfig.write_image(figname+".pdf",engine="kaleido")
  pyfig.write_image(figname+".png",engine="kaleido")

  # - offline plotly with mpl
  # pfig = tls.mpl_to_plotly(fig) 
  # plotly.offline.plot(pfig, 'plotly clone')

  app.layout= html.Div([
      dcc.Graph(id='mpl', figure=pyfig, mathjax=True),
  ])

  if live:
    app.run_server(debug=False, port=8922, host='localhost', use_reloader=False)
   
#
def paccplot(names,datas,runs,figname="fig1", live=False):
  
  # transfer to cpu  and switchback to numpy

  torch.Tensor.ndim = property(lambda self: len(self.shape))  # Fix it


  pio.renderers.default = "pdf+svg+plotly_mimetype+browser+png+vscode+colab+json"
  # pio.kaleido.scope.default_scale=1.0
  # pio.kaleido.scope.default_width=
  # pio.kaleido.scope.default_height=
  # pio.kaleido.scope.mathjax = "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js"
  app = dash.Dash()
  pyfig = go.Figure()
  
  
  for namei, dfi in zip(names,datas):
        pyfig.add_trace(go.Box(y=dfi,
                name=namei,
                # marker_color=,
                boxmean = True,
                boxpoints = "all", # outliers, suspectedoutliers, all, False
                marker_size=3,
                line_width=1,
                jitter = 0,
                whiskerwidth=0.25,
                # fillcolor = cls,
                width=0.25,
                notched=False,
                notchwidth=0,
               )
        )


  pyfig.update_layout(
      xaxis=dict(
          title="Optimizers",
          titlefont=dict(family='Open Sans'),      
      ), 
      yaxis=dict(
        #   autorange=True,
          title="Test Accuracy",
          titlefont=dict(
            #   color="red",
              family="Open Sans"
          ),
          tickfont=dict(
            #   color="red"
      )),
      width=4*100,
      height=3.5*100,
      autosize=True,
      margin=dict(l=5,r=5,b=5,t=5,pad=0
      ),
      legend=dict(
          orientation='h',
          yanchor="bottom",
          xanchor="right",
          x=1.0, y=1.02,
      ),
      # template="plotly_white", #simple_white, none
      plot_bgcolor ='white',
      font=dict(family='Open Sans',size=8,color='black'),
  )
  pyfig.add_shape(
      # Rectangle with reference to the plot
      type="rect",
      xref="paper",
      yref="paper",
      x0=0,y0=0,x1=1.0,y1=1.0,
      line=dict(
          color="black",
          width=1,
      )
  )
  pyfig.update_yaxes(
      showline=True,
      linecolor='black',
      linewidth=0,
      ticks='outside',
      tickwidth=1,
      tickcolor='black',
      # range=[-0.01,1.01],
  )
  pyfig.update_xaxes(
      showline=True,
      linecolor='black',
      linewidth=0,
      ticks='outside',
      tickwidth=1,
      tickcolor='black',
  )
  # pyfig.write_image("fig1.pdf")
  # 300dpi, width = 2inches, height = 1.5inches
  pyfig.write_image(figname+".pdf", engine="kaleido")
  pyfig.write_image(figname+".png",engine="kaleido")

  # - offline plotly with mpl
  # pfig = tls.mpl_to_plotly(fig) 
  # plotly.offline.plot(pfig, 'plotly clone')

  app.layout= html.Div([
      dcc.Graph(id='mpl', figure=pyfig, mathjax=True),
  ])

  if live:
    app.run_server(debug=False, port=8922, host='localhost', use_reloader=False)
    
#
def wilcxstatplot(names,datas,runs,figname="fig1", live=False):
  
  # transfer to cpu  and switchback to numpy

  torch.Tensor.ndim = property(lambda self: len(self.shape))  # Fix it
  
  colorscale = [[0, '#272D31'],[.5, '#d9d9d9'],[1, '#ffffff']]
  tablefig = ff.create_table(datas,colorscale=colorscale, height_constant=50)
  
  # Make text size larger
#   for i in range(len(tablefig.layout.annotations)):
#     tablefig.layout.annotations[i].font.size = 10
    
  # Make traces for graph
  tablefig.add_trace(go.Bar(x=datas['Optimizer'], y=datas['p-value'],
                      marker=dict(color='#404040'),
                      name='p-value',
                      xaxis='x2', yaxis='y2'))
  

  pio.renderers.default = "pdf+svg+plotly_mimetype+browser+png+vscode+colab+json"
  # pio.kaleido.scope.default_scale=1.0
  # pio.kaleido.scope.default_width=
  # pio.kaleido.scope.default_height=
  # pio.kaleido.scope.mathjax = "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js"
  app = dash.Dash()
  
  tablefig.update_layout(
      title_text = 'Prediction Difference: Paired Wilcoxon signed-rank test',
      xaxis=dict(
          titlefont=dict(family='Open Sans'),
          domain = [0,0.4],    
      ), 
      yaxis=dict(
          titlefont=dict(
            #   color="red",
              family="Open Sans"
          ),
          tickfont=dict(
            #   color="red"
          ),
        #   domain = [0, 0.4]
      ),
      xaxis2=dict(
          domain=[0.5,1.],
          anchor='y2'
      ),
      yaxis2=dict(title="p-values",
          titlefont=dict(
            #   color="blue",
              family="Open Sans"
          ),
          anchor="x2",
        #  domain = [0.5, 1.0]
      ),
      legend=dict(
          orientation='h',
          yanchor="bottom",
          xanchor="right",
          x=1.0, y=1.02,
      ),
      # template="plotly_white", #simple_white, none
      plot_bgcolor ='white',
      font=dict(family='Open Sans',size=8,color='black'),
      width=600,
    #   height=5*100,
    #   autosize=True,
      margin=dict(l=5,r=5,b=50,t=50,pad=0
      ),
  )
  # pyfig.write_image("fig1.pdf")
  # 300dpi, width = 2inches, height = 1.5inches
#   , width=1.5*300, height=1.5*300, 
  tablefig.write_image(figname+".pdf",engine="kaleido")
  tablefig.write_image(figname+".png",engine="kaleido")

  # - offline plotly with mpl
  # pfig = tls.mpl_to_plotly(fig) 
  # plotly.offline.plot(pfig, 'plotly clone')

  app.layout= html.Div([
      dcc.Graph(id='mpl', figure=tablefig, mathjax=True),
  ])

  if live:
    app.run_server(debug=False, port=8922, host='localhost', use_reloader=False)
    
    