#include <iostream>
#include <QtWidgets>
#include <QtNetwork>
#include <QDir>
#include "config.h"

#include "octserver.h"

// TODO: use tcpsocket timeouts to catch errors.
// TODO: use a different thread for sockets

OCTServer::OCTServer(QWidget *parent)
    : QDialog(parent)
    , statusLabel(new QLabel)
    , tcpServer(new QTcpServer)
{
    statusLabel->setTextInteractionFlags(Qt::TextBrowserInteraction);

    initServer();
    auto quitButton = new QPushButton(tr("Quit"));
    quitButton->setAutoDefault(false);

    // Connections
    connect(quitButton, &QAbstractButton::clicked, this, &QWidget::close);
    connect(tcpServer, &QTcpServer::newConnection, this, &OCTServer::slot_startConnection);
    connect(this, SIGNAL(sig_start_acquisition(int,int,int)), this, SLOT(slot_performScan(int,int,int)));
    connect(this, SIGNAL(sig_end_acquisition()), this, SLOT(slot_endConnection()));

    // Layout
    auto buttonLayout = new QHBoxLayout;
    buttonLayout->addStretch(1);
    buttonLayout->addWidget(quitButton);
    buttonLayout->addStretch(1);

    QVBoxLayout *mainLayout = nullptr;
    if (QGuiApplication::styleHints()->showIsFullScreen() || QGuiApplication::styleHints()->showIsMaximized()) {
        auto outerVerticalLayout = new QVBoxLayout(this);
        outerVerticalLayout->addItem(new QSpacerItem(0, 0, QSizePolicy::Ignored, QSizePolicy::MinimumExpanding));
        auto outerHorizontalLayout = new QHBoxLayout;
        outerHorizontalLayout->addItem(new QSpacerItem(0, 0, QSizePolicy::MinimumExpanding, QSizePolicy::Ignored));
        auto groupBox = new QGroupBox(QGuiApplication::applicationDisplayName());
        mainLayout = new QVBoxLayout(groupBox);
        outerHorizontalLayout->addWidget(groupBox);
        outerHorizontalLayout->addItem(new QSpacerItem(0, 0, QSizePolicy::MinimumExpanding, QSizePolicy::Ignored));
        outerVerticalLayout->addLayout(outerHorizontalLayout);
        outerVerticalLayout->addItem(new QSpacerItem(0, 0, QSizePolicy::Ignored, QSizePolicy::MinimumExpanding));
    } else {
        mainLayout = new QVBoxLayout(this);
    }

    mainLayout->addWidget(statusLabel);
    mainLayout->addLayout(buttonLayout);

    setWindowTitle(QGuiApplication::applicationDisplayName());
}

OCTServer::~OCTServer()
{
    tcpServer->close();
}

void OCTServer::initServer()
{
    // Set the server address and port
    QString ipAddress = OCTServer::getHostAddress();
    const QHostAddress address = QHostAddress(ipAddress);
    quint16 ipPort = SERVER_SOCKET_PORT;

    if (!tcpServer->listen(address, ipPort)) {
        QMessageBox::critical(this, tr("OCT Server"),
                              tr("Unable to start the server: %1.")
                              .arg(tcpServer->errorString()));
        close();
        return;
    }

    statusLabel->setText(tr("The server is running on\n\nIP: %1\nport: %2\n\n"
                            "Waiting for connections ...")
                         .arg(ipAddress).arg(tcpServer->serverPort()));

}

void OCTServer::slot_endConnection()
{
    std::cerr << "octserver::Closing the connection...";
    QByteArray response;
    QDataStream out(&response, QIODevice::WriteOnly);
    out << QString("OCT_done").toUtf8();
    clientConnection->write(response);
    std::cerr << " done!" << std::endl;
    //delete clientConnection; // DEBUG: the connection should be deleted when it is closed by the client.
}

void OCTServer::slot_endConnection(QString message)
{
    std::cerr << "octserver::Closing the connection...";
    QByteArray response;
    QDataStream out(&response, QIODevice::WriteOnly);
    out << message.toUtf8();
    clientConnection->write(response);
    std::cerr << " done!" << std::endl;
    //delete clientConnection; // DEBUG: the connection should be deleted when it is closed by the client.
}

QString OCTServer::getHostAddress()
{
    QString ipAddress;
    const QList<QHostAddress> ipAddressesList = QNetworkInterface::allAddresses();

    // use the first non-localhost IPv4 address
    for (const QHostAddress &entry : ipAddressesList) {
        if (entry != QHostAddress::LocalHost && entry.toIPv4Address()) {
            ipAddress = entry.toString();
            break;
        }
    }
    // if we did not find one, use IPv4 localhost
    if (ipAddress.isEmpty())
        ipAddress = QHostAddress(QHostAddress::LocalHost).toString();

    return ipAddress;
}

void OCTServer::slot_readTilePosition()
{
    QByteArray total_data, buffer;
    while(1) {
        buffer = clientConnection->read(1024);
        if (buffer.isEmpty()) {
            break;
        }
        total_data.append(buffer);
    }
    QString data = QString(total_data);
    std::cerr << data.toStdString() << std::endl;

   if ( data.startsWith("acquire_avg_fringe_intensity"))
   {
       std::cerr << "Case 1: Acquire a fringe without saving" << std::endl;
       // TODO: Acquire a fringe, compute the AUC or max intensity and send back.
       slot_endConnection(QString("4.20"));
   }
   else
   {
       std::cerr << "Case 2: Acquire and save a tile" << std::endl;
       std::cerr << "octserver::Reading tile positions: (x,y,z)=";
       // Read the tile position to acquire
       p_tile_x = data.split(" ")[0].toInt();
       p_tile_y = data.split(" ")[1].toInt();
       p_tile_z = data.split(" ")[2].toInt();

       std::cerr << "(" << p_tile_x <<"," << p_tile_y << "," << p_tile_z << ")" << std::endl;
       emit sig_start_acquisition(p_tile_x, p_tile_y, p_tile_z);
   }
}

void OCTServer::slot_startConnection()
{
    std::cerr << "octserver::Fetching the next pending connection...";
    clientConnection = tcpServer->nextPendingConnection(); // DEBUG: Try replacing by waitPendingConnection with timeout, or check if there are pending connections.
    // DEBUG: test if the returned value is nullptr (if no pending connection)

    std::cerr << " got it!" << std::endl;

    connect(clientConnection, &QIODevice::readyRead, this, &OCTServer::slot_readTilePosition);
}

void OCTServer::slot_performScan(int x, int y, int z)
{
    // Create the tile filename
    char buffer [20];
    snprintf(buffer, 20, "tile_x%02d_y%02d_z%02d", x, y, z);
    QString fileName = QString(buffer);
    emit sig_change_filename(fileName);
    // Launch a single acquisition
    emit sig_start_scan();
}